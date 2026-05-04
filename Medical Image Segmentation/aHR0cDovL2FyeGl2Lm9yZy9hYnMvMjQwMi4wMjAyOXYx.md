# ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation

Zihan Li, Yuan Zheng, Dandan Shan, Shuzhou Yang, Qingde Li, Beizhan Wang, Yuanting Zhang, Qingqi Hong, Dinggang Shen (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델을 학습시키기 위해서는 정밀한 픽셀 수준의 마스크(mask) 데이터셋이 필수적이다. 하지만 의료 영상의 수동 어노테이션(manual annotation)은 숙련된 전문가가 필요하며 시간과 비용이 매우 많이 드는 작업이다. 이를 해결하기 위해 포인트, 바운딩 박스, 스크리블(Scribble, 낙서 형태의 선)과 같은 약한 감독(weakly-supervised) 학습 방식이 제안되었다.

특히 스크리블 기반 학습은 복잡한 객체를 어노테이션 하는 데 있어 효율적이지만, 기존의 CNN 기반 프레임워크는 다음과 같은 한계가 있다.
1. **국소적 수용역(Local Receptive Field)의 한계**: CNN은 국소적인 특징 의존성만 캡처할 수 있어, 제한적인 스크리블 정보만으로는 객체의 전역적인 모양(global shape) 정보를 학습하기 어렵다.
2. **부분 활성화(Partial Activation) 문제**: Class Activation Maps(CAMs)를 통해 의사 라벨(pseudo label)을 생성할 때, 객체 전체가 아닌 가장 판별력이 높은 일부 영역만 활성화되는 경향이 있어 세그멘테이션 성능을 저하시킨다.
3. **노이즈 라벨 생성**: 불충분한 감독 정보로 인해 학습 과정에서 배경 영역에 잘못된 예측(noise)이 생성되는 문제가 빈번하게 발생한다.

본 논문의 목표는 CNN과 Transformer를 결합한 하이브리드 구조를 통해 전역적 문맥 정보를 학습하고, ACAM(Attention-guided Class Activation Map) 분기를 통해 정밀한 객체 위치를 파악함으로써 스크리블 기반 의료 영상 분할의 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN의 국소적 세부 특징(local features)과 Transformer의 전역적 표현(global representations)을 상호 보완적으로 융합하는 것이다.

1. **CNN-Transformer 하이브리드 아키텍처**: Transformer의 전역 자가 주의(global self-attention) 메커니즘을 도입하여 배경 영역의 잘못된 예측을 줄이고, CNN이 놓치기 쉬운 전역적 모양 정보를 보완한다.
2. **ACAM(Attention-guided Class Activation Map) 분기**: 전통적인 CAM의 부분 활성화 문제를 해결하기 위해 채널 및 공간 주의 변조(channel and spatial attention modulation)를 적용한 ACAM 분기를 제안한다.
3. **ACAM-Consistency Loss**: 깊은 층의 고수준 ACAM을 이용하여 얕은 층의 저수준 ACAM을 정규화함으로써, 모델이 객체 전체에 집중하도록 유도한다.
4. **혼합 감독 학습(Mixed-supervision Learning)**: 스크리블 기반의 부분 교차 엔트로피 손실, 동적으로 혼합된 의사 라벨을 이용한 pseudo-supervised 손실, 그리고 ACAM-consistency 손실을 함께 사용하여 학습 안정성을 높인다.

## 📎 Related Works

### 1. 의료 영상 분할을 위한 Transformer
최근 Vision Transformer(ViT)의 성공으로 의료 영상 분야에서도 Transformer 기반 모델(TransUNet, Swin-Unet 등)이 등장하였다. 이들은 주로 ViT를 메인 인코더로 사용하거나 CNN 뒤에 추가 인코더로 배치하여 전역적 문맥을 학습한다. 하지만 대부분의 연구가 완전 감독(fully-supervised) 또는 반감독(semi-supervised) 학습에 집중되어 있으며, 스크리블 기반의 약한 감독 학습에 Transformer를 적용한 사례는 거의 없었다.

### 2. 스크리블 기반 영상 분할 (Scribble-supervised Segmentation)
기존의 스크리블 기반 방식은 크게 두 가지로 나뉜다. 하나는 조건부 마스크 생성기와 판별기를 이용해 전역 모양을 학습하는 방식(추가적인 완전 어노테이션 마스크 필요)이고, 다른 하나는 스크리블 데이터에 직접 정교한 구조나 학습 전략을 적용하는 방식이다. 그러나 이러한 CNN 기반 방식들은 여전히 전역 정보 활용 능력이 부족하며, 의사 라벨 생성 시 발생하는 노이즈 문제에 취약하다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (ScribFormer)
ScribFormer는 **CNN 분기, Transformer 분기, ACAM 분기**의 트리플 브랜치 구조로 설계되었다. 
- **CNN 분기**: ResNet 기반의 특징 피라미드 구조를 통해 국소적 세부 특징을 추출한다.
- **Transformer 분기**: MHSA(Multi-Head Self-Attention)와 MLP 블록을 통해 전역적 표현을 학습한다.
- **ACAM 분기**: 이미지의 가장 관련성 높은 영역을 식별하여 네트워크가 집중해야 할 부분을 가이드한다.

### 2. 주요 구성 요소 및 작동 원리

#### Feature Coupling Units (FCU)
CNN의 특징 맵과 Transformer의 패치 임베딩은 차원과 공간 해상도가 다르다. FCU는 채널 및 공간 정렬(alignment) 과정을 거쳐 두 특징을 융합하여 더한다. 이 과정을 통해 CNN의 세부 패턴과 Transformer의 전역 문맥이 결합되어 각 분기의 성능을 상호 보완한다.

#### ACAM Branch
전통적인 CAM의 한계를 극복하기 위해 가우시안 변조 함수(Gaussian modulation function)를 도입하였다. 
$$f(A) = \frac{1}{\sqrt{2\pi\sigma}} e^{-\frac{(A-\mu)^2}{2\sigma^2}}$$
여기서 $A$는 공간/채널 다운샘플링을 통해 얻은 주의(attention) 값이다. 가우시안 변조는 평균 근처의 가중치를 증폭시켜 주요 특징과 관련된 영역의 중요도를 높이며, 공간 주의 변조는 사소한 활성화 영역을 복구하여 객체 전체를 포착하도록 돕는다.

#### 디코더 구조
CNN 디코더는 UNet과 유사하게 스킵 연결을 사용하며, Transformer 디코더는 전역 표현을 업샘플링하여 최종 예측값 $\text{y}_{\text{Trans}}$를 생성한다.

### 3. 학습 절차 및 손실 함수

최종 목적 함수 $L_{\text{total}}$은 다음과 같이 세 가지 손실 함수의 가중치 합으로 정의된다.
$$L_{\text{total}} = \lambda_1 \times L_{\text{ss}} + \lambda_2 \times L_{\text{pl}} + \lambda_3 \times L_{\text{acam}}$$

1. **Scribble-supervised Loss ($L_{\text{ss}}$)**: 라벨이 지정된 픽셀에 대해서만 계산하는 부분 교차 엔트로피(Partial Cross-Entropy) 손실이다.
   $$L_{\text{ce}}(y, s) = \sum_{i \in \Omega_l} \sum_{k \in K} -s_i^k \log(y_i^k)$$
   ($\Omega_l$: 라벨링 된 픽셀 집합, $s_i^k$: 스크리블 정답, $y_i^k$: 예측 확률)

2. **Pseudo-supervised Loss ($L_{\text{pl}}$)**: CNN과 Transformer의 예측 결과를 동적으로 혼합하여 생성한 하드 의사 라벨 $Y$를 사용하여 Dice 손실을 계산한다.
   $$Y = \text{argmax}(\alpha \times y_{\text{CNN}} + \beta \times y_{\text{Trans}}), \quad \alpha + \beta = 1$$
   $\alpha$는 매 반복마다 무작위로 생성되어 두 브랜치 사이의 최적의 균형을 찾도록 한다.

3. **ACAM-Consistency Loss ($L_{\text{acam}}$)**: 저수준 ACAM들이 고수준(최종 층) ACAM과 유사해지도록 강제하는 특성 수준의 일관성 손실이다.
   $$L_{\text{acam}} = \sum_{i} \omega_i \times L_{\text{ce}}(F(E_{i \dots 4}(c_i), F(c_5))$$
   여기서 $c_5$는 가장 깊은 층의 특징이며, $E$는 ACAM 인코더, $F$는 시그모이드 필터이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: ACDC (심장 MRI), MSCMRseg (LGE MRI), HeartUII (CT)의 세 가지 데이터셋을 사용하였다.
- **평가 지표**: Dice Score를 사용하여 정량적 성능을 측정하였다.
- **비교 대상**: 
    - 다양한 학습 전략을 적용한 UNet (pce, em, crf, mloss, ustr, wpce)
    - UNet++ 및 데이터 증강 기법(MixUp, CutMix, CycleMix 등)
    - 완전 감독 학습 모델 (UNet, nnUNet) 및 적대적 학습 방법(MAAG, ACCL 등)

### 2. 주요 결과
- **SOTA 대비 성능**: ScribFormer는 모든 데이터셋에서 기존의 최신 스크리블 기반 방법론인 CycleMix를 크게 상회하였다. (ACDC 기준 88.8% vs 84.8%)
- **완전 감독 모델과의 비교**: nnUNet을 제외한 대부분의 완전 감독 학습 모델보다 더 높은 성능을 보이거나 대등한 수준의 결과를 달성하였다. 이는 매우 적은 어노테이션 비용으로도 높은 성능을 낼 수 있음을 시사한다.
- **데이터 민감도**: 훈련 샘플 수가 14개로 매우 적은 상황에서도 84.7%의 준수한 정확도를 보였으며, 샘플 수가 증가함에 따라 성능이 점진적으로 향상되었다.
- **복잡도 및 효율성**: Transformer 도입으로 파라미터 수는 증가하였으나, CycleMix보다 추론 시간(inference time)이 더 빨랐다 (ScribFormer 13.96s vs CycleMix 21.21s).

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **전역 문맥의 효과**: 시각화 결과, Transformer 분기가 도입됨에 따라 배경 영역에서의 잘못된 활성화가 크게 줄어들었으며, 객체의 전체적인 형태가 Ground Truth와 훨씬 유사하게 복원되었다.
- **ACAM-Consistency의 역할**: 저수준 층의 특징들이 고수준 층의 의미론적 정보를 학습하게 함으로써, 스크리블의 희소성으로 인해 발생하는 학습 공백을 효과적으로 메웠다.
- **하이브리드 구조의 정당성**: 어블레이션 연구를 통해 CNN만 사용하거나 Transformer만 사용했을 때보다 두 구조를 융합하고 ACAM 분기를 추가했을 때 성능이 가장 높음을 확인하였다.

### 2. 한계 및 논의
- **계산 복잡도**: UNet 계열에 비해 파라미터 수가 많아 실시간 응용 분야에서는 최적화가 필요하다.
- **통계적 유의성**: 일부 SOTA 모델과의 비교에서 p-value가 0.05보다 크게 나타나는 경우가 있었는데, 이는 샘플 사이즈가 제한적이었기 때문으로 분석된다.

## 📌 TL;DR

ScribFormer는 **CNN의 국소 특징**과 **Transformer의 전역 문맥**을 융합하고, **ACAM-Consistency**를 통해 의사 라벨의 품질을 높인 최초의 Transformer 기반 스크리블 감독 의료 영상 분할 모델이다. 실험 결과, 매우 적은 양의 스크리블 어노테이션만으로도 기존 SOTA 모델 및 일부 완전 감독 모델을 능가하는 성능을 보였으며, 이는 의료 영상 분야에서 어노테이션 비용을 획기적으로 줄이면서도 고정밀 분할이 가능함을 입증한다.