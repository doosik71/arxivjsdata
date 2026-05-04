# Residual Mixture of Experts

Lemeng Wu, Mengchen Liu, Yinpeng Chen, Dongdong Chen, Xiyang Dai, Lu Yuan (2022)

## 🧩 Problem to Solve

본 논문은 Vision Transformer(ViT)의 모델 용량을 확장하기 위해 사용되는 Mixture of Experts (MoE) 구조의 높은 학습 비용 문제를 해결하고자 한다. MoE는 조건부 계산(conditional computation)과 희소성(sparsity)을 통해 파라미터 수를 획기적으로 늘리면서도 계산 효율성을 유지할 수 있게 하지만, 여전히 거대한 MoE Transformer를 처음부터 학습시키는 것은 막대한 계산 자원을 필요로 한다. 

특히 세그멘테이션(segmentation)이나 객체 검출(detection)과 같은 다운스트림 태스크는 고해상도 이미지를 입력으로 사용하므로, 모델의 너비와 깊이를 직접적으로 확장할 경우 GPU 메모리 부족 문제가 심각하게 발생한다. 따라서 본 연구의 목표는 기존의 non-MoE Transformer의 효율성을 유지하면서도, MoE의 강력한 모델 용량 이점을 누릴 수 있는 효율적인 학습 파이프라인인 Residual Mixture of Experts (RMoE)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MoE Transformer의 가중치를 입력에 독립적인 핵심 부분(input-independent core)과 입력에 종속적인 잔차 부분(input-dependent residual)으로 분해할 수 있다는 관찰에서 시작된다.

1. **가중치 분해(Weight Factorization):** MoE 전문가(expert)들의 가중치 $\theta(x)$를 $\theta(x) = \theta^0 + \theta^r(x)$로 정의한다. 여기서 $\theta^0$는 공통적인 핵심 가중치이며, $\theta^r(x)$는 개별 전문가들이 가지는 잔차 가중치이다.
2. **효율적인 학습 파이프라인 (RMoE):** 입력 독립적인 $\theta^0$는 기존의 non-MoE Transformer로 사전 학습(pretraining)하여 얻고, 입력 종속적인 $\theta^r(x)$만을 다운스트림 태스크나 중간 단계에서 효율적으로 학습시킨다.
3. **성능 보존 및 층 선택 전략:** non-MoE에서 MoE로 전환 시 발생하는 성능 저하를 막기 위한 Stop-Gradient 기반의 정렬(alignment) 기법과, 최적의 MoE 적용 위치를 찾기 위한 그래디언트 기반의 층 선택(layer selection) 방법을 제안한다.

## 📎 Related Works

### Vision Transformers
ViT를 필두로 Swin Transformer, CvT 등 계층적 구조와 대규모 사전 학습을 통해 강력한 성능을 내는 모델들이 등장하였다. 이러한 모델들은 다운스트림 태스크로의 파인튜닝(finetuning) 능력이 뛰어나지만, 모델 크기를 단순히 키우는 방식은 연산 비용의 급격한 증가를 초래한다.

### Mixture of Experts (MoE)
MoE는 입력 데이터에 따라 일부 전문가(expert)만 활성화하는 조건부 계산 방식을 통해 모델 용량을 확장한다. 최근 V-MoE와 같은 연구들이 Vision 분야에 MoE를 적용하여 SOTA 성능을 달성했으나, 여전히 수십억 개의 파라미터를 가진 모델을 학습시키는 비용은 매우 높다. 본 논문은 이러한 MoE의 용량 확장성과 non-MoE의 학습 효율성 사이의 간극을 메우고자 하며, 특히 고해상도 이미지 기반의 다운스트림 태스크에 MoE를 효율적으로 적용하는 방안을 제시한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### MoE Layer 기본 구조
MoE 층은 $n$개의 전문가 $E_i(x)$와 게이트 함수 $G(x)$로 구성된다. 입력 $x$에 대한 최종 출력 $y$는 다음과 같이 계산된다.
$$y = \sum_{i=1}^{n} G(x)_i E_i(x)$$
여기서 $G(x)$는 주로 선형 층과 softmax를 통해 구현되며, 계산 효율을 위해 Top-K 연산자를 사용하여 상위 $k$개의 전문가만 선택한다. 또한, 특정 전문가에게만 데이터가 쏠리는 현상을 막기 위해 다음과 같은 Load Balancing Loss $L(X)$를 추가한다.
$$L(X) = \left( \frac{\text{std}(\text{Imp}(X))}{\text{mean}(\text{Imp}(X))} \right)^2$$
여기서 $\text{Imp}(X)$는 배치 내에서 각 전문가가 선택된 빈도를 의미한다.

### RMoE Formulation
RMoE는 MoE의 가중치를 다음과 같이 분해한다.
$$\theta(x) = \theta^0 + \theta^r(x)$$
이 수식에서 $\theta^0$는 모든 전문가가 공유하는 핵심 가중치(core)이며, $\theta^r(x)$는 각 전문가의 고유한 잔차(residual)이다. RMoE의 학습 목표는 다음과 같은 이단계 최적화(bilevel optimization) 문제로 정의된다.
$$\min_{\theta^r} \sum L(f(x; \theta^*_0 + \theta^r(x))) \quad \text{s.t. } \theta^*_0 = \arg\min_{\theta} \sum L(f(x; \theta))$$
즉, 하위 단계에서 non-MoE 모델을 통해 $\theta^0$를 먼저 학습하고, 상위 단계에서 잔차 $\theta^r$을 학습하는 구조이다.

### 세부 설계 (Practical Designs)
1. **학습 파이프라인:** 
   - **RMoE-I (Intermediate):** 사전 학습된 non-MoE 모델 $\rightarrow$ Upstream 데이터로 짧은 MoE 중간 파인튜닝 $\rightarrow$ Downstream 파인튜닝.
   - **RMoE-D (Downstream):** 사전 학습된 non-MoE 모델 $\rightarrow$ 즉시 Downstream 파인튜닝.
2. **성능 보존 (Performance-preserving):** non-MoE 가중치를 MoE 전문가들에게 복사한 후 작은 노이즈를 추가한다. 특히, MoE 전환 초기 단계의 출력 값 감소 문제를 해결하기 위해 다음과 같은 Stop-Gradient 정렬 식을 사용한다.
   $$y = \sum_{i=1}^{n} \text{StopGrad}((1 - G(x)_i) E_i(x)) + G(x)_i E_i(x)$$
   이를 통해 Top-K 전문가만 업데이트하면서도 전체적인 출력 스케일을 유지한다.
3. **MoE 층 선택 (Layer Selection):** 모든 MLP 층을 MoE로 바꾸는 대신, Firefly Splitting 기법을 사용하여 손실 함수를 가장 많이 감소시킬 수 있는 $N$개의 핵심 층을 선택한다. 이는 테일러 근사(Taylor approximation)를 통해 각 층의 그래디언트 크기 $|s_l|$를 계산하여 결정한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 태스크:** ADE20K (시맨틱 세그멘테이션), MS-COCO (객체 검출).
- **백본 모델:** Swin-T, CvT-13, Swin-L, BeiT-L.
- **비교 대상:** Non-MoE, MoE (from scratch), RMoE-I, RMoE-D.
- **지표:** mIoU (세그멘테이션), AP (객체 검출), GPU Days (학습 비용).

### 주요 결과
1. **성능 및 비용 효율성:**
   - RMoE-I는 전체 MoE 학습과 유사한 성능을 보이면서도 학습 비용을 약 30% 절감하였다.
   - Non-MoE 대비 RMoE-I는 ADE20K에서 +1.1 / 0.9 mIoU (Swin-T/CvT-13), MS-COCO에서 +1.4 / 1.6 AP의 성능 향상을 가져왔으며, 추가 학습 비용은 3% 미만으로 매우 낮았다.
2. **대형 모델 확장성:**
   - Swin-L과 BeiT-L과 같은 거대 모델에도 RMoE를 적용한 결과, Swin-L 세그멘테이션에서 약 1.0 mIoU의 추가 이득을 얻었으며, 객체 검출에서도 AP가 상승하였다.
3. **추론 속도 및 연산량:**
   - MoE의 특성상 파라미터 수는 크게 증가하지만, 추론 시에는 소수의 전문가만 활성화되므로 FLOPs는 non-MoE와 거의 동일하며 추론 시간의 증가폭도 매우 미미함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 MoE 모델의 가중치가 특정 중심점으로 군집화된다는 시각적 분석(PCA)을 통해 RMoE의 이론적 근거를 마련하였다. 실험 결과, RMoE가 적은 자원으로도 전문가들의 특성화(Expert Specialization)를 달성했음을 확인하였는데, 이는 시각화 분석을 통해 RMoE의 전문가들이 MoE와 유사하게 배경/전경 등 서로 다른 이미지 특성에 반응함을 보여주어 입증되었다.

### 한계 및 논의사항
RMoE-D(직접 다운스트림 학습)가 RMoE-I보다 성능이 다소 낮은 이유는 Load Balancing Loss의 영향으로 분석된다. 다운스트림 태스크의 새로운 디코더 헤드와 큰 학습률이 전문가 간의 균형을 깨뜨리며, 이를 강제로 맞추려 하면 오히려 성능이 저하되는 트레이드오프가 존재한다. 또한, 전문가 수 $n$을 16개 이상으로 늘렸을 때 성능 향상이 뚜렷하지 않았는데, 이는 현재의 Load Balancing Loss 방식이 과도한 전문가 수 상황에서 개별 전문가의 성능을 저해할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 Vision Transformer를 효율적으로 확장하기 위한 **Residual Mixture of Experts (RMoE)** 파이프라인을 제안한다. 전문가의 가중치를 **'공통 핵심($\theta^0$) + 개별 잔차($\theta^r$)'**로 분해하여, 이미 학습된 non-MoE 모델을 핵심으로 사용하고 잔차만을 학습시킴으로써 **학습 비용을 획기적으로 줄이면서도 대형 MoE 모델의 성능에 근접**하는 성과를 거두었다. 이는 거대 모델 학습 자원이 부족한 환경에서도 기존의 프리트레인된 모델을 기반으로 효율적인 모델 확장이 가능함을 보여준 연구이다.