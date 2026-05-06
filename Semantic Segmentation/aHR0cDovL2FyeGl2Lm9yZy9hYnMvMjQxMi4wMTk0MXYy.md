# Global Average Feature Augmentation for Robust Semantic Segmentation with Transformers

Alberto G. Rodriguez Salgado, Maying Shen, Philipp Harzig, Peter Mayer, Jose M. Alvarez (2024)

## 🧩 Problem to Solve

본 논문은 시맨틱 세그멘테이션(Semantic Segmentation) 모델, 특히 Vision Transformer(ViT) 기반 모델들이 학습 데이터와 다른 분포를 가진 Out-of-distribution(OOD) 데이터, 즉 블러(blur)나 노이즈(noise)와 같은 시각적 오염(visual corruptions)이 포함된 이미지에 대해 취약하다는 문제를 해결하고자 한다.

현대적인 딥러닝 모델의 실제 배포를 위해서는 이러한 오염된 데이터에 대한 강건성(Robustness) 확보가 필수적이다. 기존의 이미지 공간 증강(Image-space augmentation) 기법인 AugMix 등은 효과적이지만, 학습 시 계산 비용을 크게 증가시키는 단점이 있다. 또한, 특징 공간 증강(Feature-space augmentation) 기법인 Stochastic Feature Augmentation(SFA)은 모든 특징에 대해 독립적인 가우시안 노이즈를 추가하며 클래스별 공분산 행렬(covariance matrix)을 계산해야 하므로, 대규모 데이터셋에서는 연산 효율성이 매우 떨어진다는 문제가 있다.

따라서 본 연구의 목표는 계산 오버헤드를 최소화하면서도 Vision Transformer의 강건성을 유의미하게 향상시킬 수 있는 효율적인 특징 증강 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Channel-Wise Feature Augmentation (CWFA)**라는 단순하고 효율적인 특징 증강 모듈을 제안한 것이다.

CWFA의 중심적인 직관은 이미지 오염(예: 노이즈)이 공간적 위치와 관계없이 동일한 채널 블록 내의 모든 특징에 유사한 영향을 미친다는 점이다. 이를 위해 각 특징마다 독립적인 섭동(perturbation)을 주는 대신, 해당 인코더의 전역 평균 특징(Global Average Feature)을 기반으로 채널별 섭동 벡터를 생성하여 모든 공간 위치에 균일하게 적용한다. 이 방식은 Global Average Pooling(GAP) 연산을 통해 매우 빠르게 계산될 수 있으며, 모델의 크기나 아키텍처에 상관없이 플러그인 형태로 적용 가능하다는 장점이 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개한다.

1. **Vision Transformer의 강건성**: SegFormer, Swin Transformer, Twins 등이 세그멘테이션 작업에서 우수한 성능을 보였으며, 특히 SegFormer는 CNN 기반 모델보다 강건함이 입증되었다. 최근 FAN 및 FAN+STL과 같은 연구들이 Transformer 기반 모델의 일반화 능력을 더욱 향상시키려 시도하였다.
2. **이미지 공간 증강**: Cutout, Mixup, CutMix 및 최신 기법인 AugMix, PixMix 등이 제안되었다. 이러한 방법들은 강건성을 높이지만, 계산 비용이 높고 모든 종류의 오염에 일반화되지 않는 경향이 있다.
3. **특징 공간 증강**: Noisy Feature Mixup 및 Stability Training 등이 있으며, 특히 SFA(Stochastic Feature Augmentation)가 CWFA와 가장 유사하다.

**SFA와의 차별점**:

- **섭동의 종류**: SFA는 독립적인 가우시안 노이즈를 사용하는 반면, CWFA는 정규화된 특징 자체를 섭동으로 사용한다.
- **적용 방식**: SFA는 특징 맵의 모든 픽셀($H \times W \times C$)에 독립적인 노이즈를 주지만, CWFA는 채널 차원($C$)에 대해서만 섭동을 계산하여 공간 차원($H \times W$) 전체에 동일하게 적용한다. 결과적으로 CWFA가 훨씬 효율적이며 더 높은 강건성 이득을 제공한다.

## 🛠️ Methodology

### 전체 파이프라인

CWFA는 인코더-디코더 구조의 세그멘테이션 모델 $f$에서 각 인코더 $\text{enc}_i$ 뒤에 위치하는 플러그인 모듈이다. 학습 시 특정 확률 $p_{\text{augm}}$에 따라 특징 맵에 섭동을 추가하여 모델이 오염된 특징 분포에 적응하도록 유도한다.

### 상세 방법 및 방정식

특정 인코더 $i$에서 출력된 특징 맵을 $X_i \in \mathbb{R}^{C_i \times H_i \times W_i}$라고 할 때, CWFA는 다음과 같은 단계로 동작한다.

1. **전역 채널 평균 특징 계산**:
    특징 맵의 공간 차원을 평균 내어 대표 특징 벡터 $x_i \in \mathbb{R}^{C_i}$를 구한다.
    $$x_{i,c} = \frac{1}{H_i W_i} \sum_{j=1}^{H_i} \sum_{m=1}^{W_i} X_{i,(c,j,m)}$$

2. **특징 섭동 벡터 생성**:
    구해진 평균 특징을 $L_2$ 정규화하고 섭동 강도 $\epsilon$을 곱하여 섭동 벡터 $p$를 생성한다.
    $$p = \epsilon \frac{x_i}{\|x_i\|_2}$$

3. **특징 증강 적용**:
    원래의 특징 맵에 생성된 섭동 벡터를 더하여 증강된 특징 $\hat{X}_i$를 생성한다.
    $$\hat{X}_{i,(c,j,m)} = X_{i,(c,j,m)} + p_c$$

여기서 $\epsilon$은 섭동의 강도를 조절하며, 값이 클수록 더 강한 섭동이 적용된다.

### 구현 세부사항

- **적용 대상**: SegFormer, Swin Transformer, Twins의 모든 인코더 단계에 적용한다.
- **하이퍼파라미터**: 증강 확률 $p_{\text{augm}} = 0.3$을 사용하며, 모델 크기에 따라 $\epsilon$을 다르게 설정한다 (소형 모델 B0~B2는 $\epsilon=9$, 대형 모델 B3~B5는 $\epsilon=15$).
- **학습 전략**: 모델이 의미 있는 표현을 먼저 학습할 수 있도록, 초기 16k 이터레이션 이후부터 CWFA를 적용하는 것이 가장 효과적임을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cityscapes, ADE20K 및 이들의 오염 버전인 Cityscapes-C, Cityscapes-$\bar{C}$, ADE20K-C를 사용한다.
- **지표**: mIoU 및 Retention Rate ($R = \frac{\text{Robust mIoU}}{\text{Clean mIoU}}$)를 측정한다.
- **비교 대상**: Baseline SegFormer, AugMix, SFA, PixMix, 그리고 SOTA 모델인 FAN, STL 등이 비교 대상이다.

### 주요 결과

1. **강건성 향상**: Cityscapes-C에서 SegFormer-B1 모델의 경우, CWFA 적용 시 Impulse Noise에 대해 mIoU가 최대 27.7% 향상되었다.
2. **SOTA 달성**: SegFormer-B5에 CWFA를 적용했을 때, Cityscapes-C 벤치마크에서 84.3%의 Retention Rate를 기록하며 새로운 State-of-the-art(SOTA)를 달성하였다. 이는 기존 FAN+STL 대비 0.7% 향상된 수치이다.
3. **일반화 성능**: Cityscapes-C와 완전히 다른 오염원들로 구성된 Cityscapes-$\bar{C}$에서도 강건성이 크게 향상되어, 제로샷(zero-shot) 강건성이 입증되었다.
4. **효율성**: 학습 시간 측정 결과, AugMix는 학습 시간을 47% 증가시킨 반면, CWFA는 단 2%의 오버헤드만 발생시켰다.
5. **아키텍처 범용성**: SegFormer 외에도 Twins(PCPVT, SVT)와 Swin-T 모델 모두에서 강건성이 향상되었다. 특히 전역 어텐션(Global Attention)을 사용하는 모델에서 더 큰 효과가 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **ViT의 특성 활용**: 본 연구는 Vision Transformer의 전역 어텐션 메커니즘이 이미지 전체의 정보를 캡슐화하고 있음을 이용하여, GAP를 통한 전역 섭동 계산이 매우 효과적임을 보여주었다.
- **모델 크기와의 상관관계**: 소형 모델(B0~B2)이 대형 모델보다 CWFA를 통한 강건성 이득이 더 크게 나타났다. 이는 소형 모델이 본래 오염에 더 취약하기 때문이며, CWFA가 이를 효과적으로 보완함을 시사한다.
- **구성 요소의 중요성**: Ablation study를 통해 GAP 연산을 제거하거나 가우시안 노이즈를 사용할 경우 성능이 크게 하락함을 확인하였다. 즉, **'채널별 전역 평균'**과 **'정규화된 특징 기반 섭동'**이 CWFA의 핵심 성공 요인이다.

### 한계 및 논의사항

- **$\epsilon$ 설정의 민감도**: $\epsilon$ 값에 따라 성능 변화가 있으며, 모델 크기에 따라 다른 값을 설정해야 한다. 하지만 저자들은 특정 범위 내에서는 성능 변화가 크지 않아 하이퍼파라미터 탐색이 치명적이지는 않다고 주장한다.
- **초기 학습의 필요성**: 학습 초기 단계부터 섭동을 주면 모델이 의미 있는 표현을 학습하는 데 방해가 되어 오히려 성능이 저하될 수 있다. 따라서 일정 기간의 'Burn-in' 학습이 필요하다는 점은 실무 적용 시 주의해야 할 점이다.

## 📌 TL;DR

본 논문은 Vision Transformer 기반 시맨틱 세그멘테이션 모델의 강건성을 높이기 위해, 전역 평균 특징을 이용해 채널별 섭동을 주는 **CWFA(Channel-Wise Feature Augmentation)** 기법을 제안하였다. 이 방법은 계산 비용을 거의 증가시키지 않으면서도(학습 시간 +2%), 다양한 오염 데이터에 대한 강건성을 획기적으로 높여 SegFormer-B5 기준 SOTA Retention Rate를 달성하였다. 특히 이미지 공간 증강보다 훨씬 효율적이며, 다양한 ViT 아키텍처와 데이터셋에 범용적으로 적용 가능하여 향후 실환경 자율주행 등 강건성이 필수적인 분야에 중요한 기여를 할 것으로 기대된다.
