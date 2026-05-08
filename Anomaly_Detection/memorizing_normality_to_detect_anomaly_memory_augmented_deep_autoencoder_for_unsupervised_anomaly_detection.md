# Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection

Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, Anton van den Hengel

## 🧩 Problem to Solve

기존 딥 오토인코더(AE) 기반의 비지도 이상 감지(unsupervised anomaly detection) 방법론은 정상 데이터만을 학습하여 이상 데이터에 대해 높은 재구성 오류(reconstruction error)를 보일 것이라는 가정에 기반합니다. 그러나 오토인코더가 때로는 이상 데이터까지도 잘 "일반화(generalize)"하여 낮은 재구성 오류를 생성하는 경우가 발생하며, 이로 인해 이상치를 놓치는 오탐지 문제가 발생할 수 있습니다. 이는 특히 고차원 데이터에서 더욱 두드러집니다.

## ✨ Key Contributions

- **메모리 증강 오토인코더(MemAE) 제안:** 오토인코더에 메모리 모듈을 추가하여 이상 감지 성능을 향상시켰습니다.
- **정상성 기억(Memorizing Normality) 메커니즘:** 메모리 모듈이 훈련 단계에서 정상 데이터의 프로토타입(prototypical) 패턴을 학습하고 기억하도록 설계하여, 이상 데이터 입력 시 재구성 오류를 증폭시킵니다.
- **희소 어드레싱(Sparse Addressing) 기법:** 메모리 어드레싱 과정에서 차등 가능한(differentiable) 하드 수축(hard shrinkage) 연산자와 엔트로피 손실(entropy loss)을 활용하여, 입력 인코딩과 가장 유사한 소수의 메모리 아이템만을 선택적으로 사용함으로써 희소성을 유도합니다.
- **다양한 데이터 유형에 대한 일반화:** 이미지, 비디오, 사이버 보안 데이터 등 다양한 응용 분야에서 MemAE의 뛰어난 일반화 능력과 효과성을 입증했습니다.

## 📎 Related Works

- **이상 감지(Anomaly Detection):**
  - 단일 클래스 분류(one-class classification) 방법: One-class SVM, 딥 원-클래스 네트워크.
  - 재구성 기반(reconstruction-based) 방법: PCA, 희소 표현(sparse representation), 딥 오토인코더(AE, VAE, DSEBM).
- **비디오 이상 감지(Video Anomaly Detection):** MPPCA, MDT, 희소 코딩(sparse coding), 3D 합성곱(3D convolution) 기반 AE, Stacked RNN, 프레임 예측 네트워크 등.
- **메모리 네트워크(Memory Networks):** 뉴럴 튜링 머신(Neural Turing Machines), 원샷 학습(one-shot learning)을 위한 메모리 네트워크, 멀티모달 데이터 생성 등에 활용된 외부 메모리 사용.

## 🛠️ Methodology

MemAE는 인코더, 디코더, 그리고 메모리 모듈로 구성됩니다.

1. **인코딩 및 쿼리 생성:** 입력 $x$는 인코더 $f_e(\cdot)$를 통해 인코딩 $z = f_e(x; \theta_e)$로 변환됩니다. 이 $z$는 메모리 모듈에 대한 쿼리(query)로 사용됩니다.
2. **메모리 기반 표현:** 메모리 $M \in R^{N \times C}$는 $N$개의 메모리 아이템 $m_i$로 구성됩니다. 각 $m_i$는 정상 데이터의 대표적인 인코딩 패턴을 저장합니다.
3. **어텐션 기반 희소 어드레싱:**
   - 쿼리 $z$와 각 메모리 아이템 $m_i$ 간의 코사인 유사도 $d(z, m_i)$를 계산합니다.
   - 이 유사도에 기반하여 소프트 어드레싱 가중치 $w_i$를 소프트맥스(softmax)를 통해 얻습니다:
     $$w_i = \frac{\exp(d(z,m_i))}{\sum_{j=1}^{N} \exp(d(z,m_j))}$$
   - **하드 수축(Hard Shrinkage) 연산:** 희소성을 강화하기 위해, 임계값 $\lambda$보다 작은 $w_i$를 0으로 만드는 하드 수축 연산 $h(w_i; \lambda)$를 적용하여 $\hat{w}_i$를 얻습니다. 이는 연속적인 ReLU 활성화 함수를 사용하여 미분 가능하게 구현됩니다:
     $$\hat{w}_i = \frac{\max(w_i - \lambda, 0) \cdot w_i}{|w_i - \lambda| + \epsilon}$$
   - $\hat{w}$는 다시 정규화됩니다: $\hat{w}_i = \hat{w}_i / \|\hat{w}\|_1$.
4. **재구성:** 정규화된 희소 가중치 $\hat{w}$와 메모리 $M$을 사용하여 새로운 잠재 표현 $\hat{z} = \hat{w}M$을 생성하고, 디코더 $f_d(\cdot)$를 통해 입력 $x$를 재구성합니다: $\hat{x} = f_d(\hat{z}; \theta_d)$.
5. **훈련 목표:** 훈련 시에는 재구성 오류 $R(x_t, \hat{x}_t) = \|x_t - \hat{x}_t\|_2^2$와 어드레싱 가중치 $\hat{w}_t$의 엔트로피 손실 $E(\hat{w}_t) = \sum_{i=1}^{N} -\hat{w}_i \cdot \log(\hat{w}_i)$를 결합한 목적 함수를 최소화합니다:
   $$L(\theta_e, \theta_d, M) = \frac{1}{T}\sum_{t=1}^{T} (R(x_t, \hat{x}_t) + \alpha E(\hat{w}_t))$$
   이를 통해 메모리는 정상 데이터의 프로토타입 패턴을 효율적으로 기록하게 됩니다.
6. **테스트 단계:** 학습된 메모리는 고정되며, 재구성 오류 $e = \|x - \hat{x}\|_2^2$를 이상 감지 기준으로 사용합니다. 이상 샘플의 경우, 인코딩이 메모리의 정상 패턴으로 대체되어 재구성 오류가 크게 증가합니다.

## 📊 Results

- **이미지 데이터 (MNIST, CIFAR-10):**
  - MemAE는 OC-SVM, KDE, VAE, PixCNN, DSEBM 등 기존 방법과 메모리 없는 AE 및 희소성 없는 MemAE 변형(MemAE-nonSpar)보다 높은 AUC 값을 달성하여 우수한 성능을 보였습니다.
  - 특히 MNIST 데이터셋에서 MemAE는 0.9751의 평균 AUC를, CIFAR-10에서는 0.6088을 기록했습니다.
  - 시각화 결과, MemAE는 훈련된 정상 패턴(예: 숫자 '5')만을 기억하여 비정상 입력(예: 숫자 '9')을 정상 패턴으로 재구성하여 재구성 오류를 크게 부각시켰습니다.
- **비디오 데이터 (UCSD-Ped2, CUHK Avenue, ShanghaiTech):**
  - MemAE는 AE-Conv2D/3D, TSC, StackRNN 등 최신 비디오 이상 감지 방법론 대비 동등하거나 더 우수한 AUC 성능을 보였습니다.
  - UCSD-Ped2에서 0.9410 AUC를 달성하여 기존 SOTA 모델에 비해 성능이 향상되었습니다.
  - 정상성 점수(normality score)가 이상 이벤트 발생 시 즉시 감소하는 것을 확인했습니다.
  - 메모리 크기에 대한 강건성(robustness)을 보여, 충분히 큰 메모리 크기에서는 안정적인 성능을 유지했습니다.
  - 프레임당 0.0262초의 빠른 처리 속도로 실시간 적용 가능성을 입증했습니다.
- **사이버 보안 데이터 (KDDCUP99):**
  - MemAE는 OC-SVM, DCN, DSEBM, DAGMM 등 비교 방법들 중에서 가장 높은 정밀도(precision), 재현율(recall), F1 점수를 달성했습니다 (F1 0.9641).
  - 이는 컴퓨터 비전 외의 분야에서도 MemAE의 일반화 능력을 증명합니다.
- **어블레이션 스터디(Ablation Studies):** 하드 수축 연산과 엔트로피 손실 모두 희소성을 유도하는 데 중요한 역할을 하며, 둘 중 하나라도 제거하면 성능 저하가 발생함을 확인했습니다.

## 🧠 Insights & Discussion

- **강화된 이상 감지:** MemAE는 메모리 모듈을 통해 정상 데이터의 핵심적인 특징만을 명시적으로 "기억"함으로써, 이상 데이터가 입력되었을 때 의도적으로 정상 패턴으로 재구성하게 만듭니다. 이로 인해 이상 데이터와 재구성된 정상 데이터 사이의 차이가 증폭되어 재구성 오류를 이상 감지 지표로 더욱 신뢰할 수 있게 만듭니다.
- **일반화 능력:** 인코더-디코더 구조에 구애받지 않고 메모리 모듈을 추가하는 방식이므로, 이미지, 비디오, 시계열 데이터 등 다양한 데이터 유형에 적용될 수 있습니다.
- **희소성의 중요성:** 희소 어드레싱은 모델이 적은 수의 관련 메모리 아이템만을 사용하여 효율적이고 정보적인 표현을 학습하도록 유도하며, 이상 샘플이 조밀한(dense) 어드레싱 가중치로 인해 잘 재구성되는 것을 방지합니다.
- **미래 연구 방향:** 메모리 어드레싱 가중치를 이상 감지 기준으로 활용하거나, 더 복잡한 기본 모델에 메모리 모듈을 통합하여 더욱 도전적인 애플리케이션에 적용하는 연구가 가능합니다.

## 📌 TL;DR

기존 오토인코더가 이상 데이터를 잘 재구성하여 이상 감지에 실패하는 문제를 해결하기 위해, 본 논문은 메모리 증강 오토인코더(MemAE)를 제안합니다. MemAE는 인코딩된 입력을 쿼리로 사용하여 메모리에서 가장 관련성 높은 정상 패턴을 검색하고, 희소 어드레싱 기법을 통해 이 패턴들로만 입력을 재구성합니다. 이는 정상 샘플은 잘 재구성하고 이상 샘플의 재구성 오류를 크게 증폭시켜 이상 감지 성능을 향상시킵니다. 이미지, 비디오, 사이버 보안 데이터셋에서의 실험을 통해 MemAE의 뛰어난 일반화 능력과 효과성을 입증했습니다.
