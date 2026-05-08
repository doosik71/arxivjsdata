# Few-shot Adaptation of Medical Vision-Language Models

Fereshteh Shakeri, Yunshi Huang, Julio Silva-Rodríguez, Houda Bahig, An Tang, Jose Dolz, and Ismail Ben Ayed (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야에서 Vision-Language Models (VLMs)를 매우 제한된 수의 학습 데이터(Few-shot)만으로 효율적으로 적응(Adaptation)시키는 문제를 해결하고자 한다.

의료 분야에서는 신뢰할 수 있는 모델을 학습시키기 위해 대규모의 레이블링된 데이터셋이 필요하지만, 실제로 전문의에 의해 어노테이션된 데이터는 매우 희소하다. 또한, 스캐너의 종류, 염색 방식, 인구통계학적 차이로 인한 Domain Drift가 빈번하게 발생하므로, 적은 수의 샘플만으로 모델을 빠르게 조정하는 Few-shot Adaptation 기술이 필수적이다.

기존의 의료 VLM 연구들은 주로 전체 데이터셋의 1% 또는 10%를 사용하여 파인튜닝을 진행했는데, 이는 여전히 수백에서 수천 개의 샘플을 필요로 하므로 희귀 질환과 같은 상황에서는 적용하기 어렵다. 또한, 전체 인코더를 파인튜닝하는 방식은 계산 비용이 매우 높고, 의료 데이터의 개인정보 보호 문제로 인해 모델의 내부 가중치에 접근할 수 없는 Black-box 설정에서의 적응 방법이 필요하다는 점이 주요 문제로 제기된다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 VLM의 Few-shot 적응을 위한 체계적인 벤치마크를 구축하고, 매우 단순하면서도 강력한 적응 전략인 **LP+text**를 제안한 것이다.

중심적인 아이디어는 기존의 Linear Probing (LP)이 텍스트 인코더의 지식을 완전히 무시한다는 점에 착안하여, 시각적 프로토타입(Visual Prototypes)과 텍스트 임베딩(Text Embeddings)을 학습 가능한 클래스별 가중치(Class-wise multipliers)를 통해 최적으로 결합(Blending)하는 것이다. 이를 통해 복잡한 Prompt-learning이나 Adapter 기반 방식보다 훨씬 적은 계산 비용으로 경쟁력 있는 성능을 달성하며, 텍스트 인코더의 출력값만 사용하므로 Black-box 설정에서도 작동 가능하다.

## 📎 Related Works

논문은 Few-shot 적응을 위한 기존 접근 방식을 크게 두 가지 범주로 설명한다.

1. **Prompt Learning**: CoOp, CoCoOp, KgCoOp 등이 대표적이다. 텍스트 입력을 학습 가능한 연속 벡터(Continuous vectors)로 최적화하여 타겟 태스크에 맞춘다. 하지만 이 방식은 텍스트 인코더 전체에 대해 그래디언트 역전파(Back-propagation)를 수행해야 하므로 계산 및 메모리 오버헤드가 매우 크며, 텍스트 인코더의 파라미터에 접근해야 한다는 한계가 있다.
2. **Black-box Adapters**: CLIP-Adapter, Tip-Adapter 등이 있으며, 프리트레인된 시각 및 텍스트 특징에 비선형 변환을 적용한다. 텍스트 인코더를 직접 수정하지 않으므로 효율적이지만, 시각 특징과 텍스트 특징을 결합하는 하이퍼파라미터 설정에 매우 민감하여 많은 양의 Grid Search가 필요하다는 단점이 있다.

본 논문이 제안하는 방식은 이러한 복잡성을 제거하고, 단순한 선형 결합의 일반화 형태를 통해 효율성과 성능을 동시에 확보하여 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 흐름

본 연구는 대규모 이미지-텍스트 쌍으로 사전 학습된 Foundation Model을 기반으로 한다. 타겟 데이터셋에서 클래스당 매우 적은 수의 샘플($S \in \{1, 2, 4, 8, 16\}$)만 사용하여 모델을 적응시킨 후, 테스트 세트의 레이블을 예측하는 구조이다.

### 주요 구성 요소 및 방정식

1. **특징 추출**:
   - 시각 인코더 $\theta_v$를 통해 이미지 $x_i$로부터 시각 임베딩 $f_i = \theta_v(x_i)$를 얻는다.
   - 텍스트 인코더 $\theta_t$를 통해 클래스 설명 $z_k$로부터 텍스트 임베딩 $t_k = \theta_t(z_k)$를 얻는다.
   - 이때, 두 인코더 모두 동결(Frozen) 상태로 유지된다.

2. **Standard Linear Probe (LP)**:
   시각 특징만을 사용하여 선형 분류기를 학습시킨다. 손실 함수는 다음과 같다.
   $$L_{CE}(w) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \ln p_{ik}(w)$$
   여기서 확률 $p_{ik}(w)$는 다음과 같이 계산된다.
   $$p_{ik}(w) = \frac{\exp(f_i^T w_k)}{\sum_{j=1}^{K} \exp(f_i^T w_j)}$$
   ($w_k$는 학습 가능한 클래스 프로토타입이다.)

3. **Proposed Text-driven Linear Probe (LP+text)**:
   LP의 한계를 극복하기 위해 텍스트 임베딩 $t_k$를 결합하고, 이를 조절하는 학습 가능한 multiplier $\alpha_k$를 도입한다.
   $$p_{ik}(w, \alpha) = \frac{\exp(f_i^T (w_k + \alpha_k t_k))}{\sum_{j=1}^{K} \exp(f_i^T (w_j + \alpha_j t_j))}$$
   손실 함수 $L_{CE}(w, \alpha)$를 통해 시각적 프로토타입 $w$와 결합 파라미터 $\alpha$를 동시에 최적화한다.

### 학습 절차

- **최적화**: Full-batch gradient descent를 사용하며, Lipschitz-gradient 특성을 이용해 step size를 암시적으로 결정하는 효율적인 최적화 알고리즘을 적용한다. 이를 통해 학습률(Learning rate)에 대한 과도한 튜닝 과정을 생략할 수 있다.
- **Black-box 설정**: 텍스트 인코더 $\theta_t$의 내부 파라미터는 수정하지 않고, 오직 출력된 임베딩 $t_k$만을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 모델**:
  - **Histology**: Quilt-1M 모델 사용 / NCT-CRC, SICAPv2, SkinCancer 데이터셋.
  - **Ophthalmology**: FLAIR 모델 사용 / MESSI-DOR, FIVES, ODIR200x3 데이터셋.
  - **Radiology**: MedCLIP 모델 사용 / CheXpert, MIMIC-CXR, RSNA 데이터셋.
- **지표**: Balanced Average Accuracy (ACA)를 사용하며, 5개의 랜덤 시드로 평균을 측정하였다.
- **비교 대상**: Zero-shot, Prompt Learning (CoOp, CoCoOp, KgCoOp), Adapters (CLIP-Adapter, Tip-Adapter-F), Standard LP.

### 주요 결과

1. **정량적 성능**:
   - LP+text는 대부분의 벤치마크에서 Prompt Learning 방식보다 훨씬 높은 성능을 보였으며, Adapter 기반 방식들과 대등하거나 더 우수한 성능을 기록하였다.
   - 특히 데이터가 극도로 적은 $S=1$ 상황에서 표준 LP는 성능이 크게 하락하지만, LP+text는 텍스트 정보를 활용함으로써 이 하락을 방지하였다.
2. **계산 효율성**:
   - **학습 시간**: LP+text는 약 4초로 가장 빨랐다. (CoCoOp는 12분, CoOp/KgCoOp는 3분 소요)
   - **메모리 사용량**: LP+text는 약 800MB의 GPU 메모리를 사용한 반면, CoCoOp는 최대 28GB를 사용하여 저사양 환경에서의 적용 가능성을 입증하였다.
   - **파라미터 수**: 학습해야 할 파라미터 수가 매우 적어 모델 복잡도가 낮다.

## 🧠 Insights & Discussion

### 강점

본 논문은 복잡한 아키텍처의 추가 없이 단순한 선형 결합과 가중치 최적화만으로도 의료 VLM의 Few-shot 적응이 가능하다는 것을 보여주었다. 특히 의료 현장에서 매우 중요한 **Black-box 접근 가능성**과 **극심한 저사양 계산 환경**에서의 효율성을 동시에 확보했다는 점이 큰 강점이다.

### 한계 및 논의사항

- **가정**: 본 연구는 프리트레인된 Foundation Model의 임베딩 공간이 이미 어느 정도 정렬되어 있다는 가정 하에 작동한다. 만약 모델의 기본 성능(Zero-shot)이 매우 낮다면, 단순히 임베딩을 결합하는 방식만으로는 한계가 있을 수 있다.
- **하이퍼파라미터**: LP+text는 튜닝 과정을 최소화했으나, 여전히 $\alpha$와 $w$의 초기값이나 최적화 알고리즘의 수렴 속도가 성능에 영향을 줄 수 있다.

### 비판적 해석

Prompt Learning이 최근 CV 분야에서 각광받고 있음에도 불구하고, 의료 분야에서는 그 계산 비용 대비 성능 이득이 적다는 결과가 흥미롭다. 이는 의료 영상의 특성이 자연 영상보다 더 fine-grained 하여, 텍스트 프롬프트를 정교하게 다듬는 것보다 시각적 특징과 전문 지식(텍스트 임베딩)을 직접적으로 결합하는 방식이 더 효과적일 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 의료 VLM을 위한 최초의 엄격한 Few-shot 적응 벤치마크를 제안하고, 시각 프로토타입과 텍스트 임베딩을 학습 가능한 가중치로 결합하는 **LP+text** 방법을 제시하였다. 이 방법은 복잡한 Prompt Learning이나 Adapter보다 훨씬 빠르고 메모리 효율적이며, Black-box 설정에서도 경쟁력 있는 성능을 보여 의료 현장의 실질적인 배포 가능성을 높였다.
