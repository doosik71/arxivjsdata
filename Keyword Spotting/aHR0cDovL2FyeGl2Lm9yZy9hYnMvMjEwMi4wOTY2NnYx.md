# DYNAMIC CURRICULUM LEARNING VIA DATA PARAMETERS FOR NOISE ROBUST KEYWORD SPOTTING

Takuya Higuchi, Shreyas Saxena, Mehrez Souden, Tien Dung Tran, Masood Delfarah and Chandra Dhir (2021)

## 🧩 Problem to Solve

본 논문은 Keyword Spotting(KWS) 시스템의 소음 강건성(Noise Robustness)을 높이는 문제를 다룬다. 특히 원거리 음성 인식 시나리오에서 소음은 인식 성능을 저하시키는 결정적인 요인이 된다. 이를 해결하기 위해 깨끗한 음성 데이터에 인위적으로 소음을 섞는 Multicondition Training(데이터 증강) 기법이 널리 사용되지만, 학습 데이터의 난이도(예: SNR 수준)가 매우 다양하기 때문에 어떤 수준의 난이도가 모델의 일반화 성능을 최대화하는지 알기 어렵다는 문제가 있다.

결과적으로, 단순히 모든 데이터를 한꺼번에 학습시키는 것보다 효율적인 학습 순서인 Curriculum Learning을 설계하는 것이 중요하지만, 각 샘플의 난이도를 수동으로 정의하거나 레이블링하여 체계적인 학습 스케줄을 짜는 것은 매우 어렵다. 따라서 본 논문의 목표는 추가적인 어노테이션 없이도 학습 데이터의 난이도를 자동으로 학습하여 최적의 학습 경로를 찾아가는 Dynamic Curriculum Learning 기법을 KWS에 적용하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 처리 분야에서 제안된 **Data Parameters** 개념을 음향 모델링(Acoustic Modeling)에 도입하는 것이다.

핵심 직관은 모델 파라미터와 함께 최적화되는 별도의 '데이터 파라미터'를 도입하여, 각 클래스(Class)와 개별 샘플(Instance)의 Logits을 스케일링하는 것이다. 이를 통해 모델이 학습 초기에 상대적으로 쉬운 샘플에 집중하고, 학습이 진행됨에 따라 점진적으로 어려운 샘플을 학습하도록 유도하는 자동화된 Curriculum Learning을 구현한다. 특히, 이 과정에서 SNR과 같은 난이도 지표를 명시적으로 제공하지 않고도 Gradient Descent 최적화 과정 속에서 데이터의 중요도를 스스로 조절하게 만든다.

## 📎 Related Works

기존의 소음 강건성 향상 방법으로는 전단부의 Speech Enhancement나 단순한 데이터 증강(Multicondition Training)이 있다. 특히 Sivasankaran et al. [29]은 Multicondition Training을 위해 발화(Utterance) 단위의 가중치 파라미터를 도입하여 Cross Entropy Loss를 직접 스케일링하는 방식을 제안하였다.

그러나 기존 방식은 SNR을 기준으로 데이터를 나누고 서브셋 단위로 가중치를 적용했으며, 발화 단위(Utterance-wise)로 가중치를 적용할 경우 오버피팅이 발생할 수 있다고 지적하였다. 반면, 본 논문에서 제안하는 방식은 Logits 단계에서 스케일링을 수행하며, 추가적인 SNR 레이블 없이도 개별 발화 단위의 데이터 파라미터를 최적화함으로써 오버피팅 문제를 완화하고 성능을 향상시켰다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 구조 및 데이터 파라미터 정의

본 논문은 DNN 기반의 음향 모델을 사용하며, 입력 데이터 $x^i_t$에 대해 모델 $f_\theta$가 Logits $z^i_t$를 출력한다. 여기서 $i$는 발화 인덱스, $t$는 시간 프레임, $\theta$는 모델 파라미터이다. 제안된 방법은 다음과 같은 두 가지 파라미터를 도입한다.

1. **Class Parameters ($\sigma^{class}$):** 타겟 클래스(예: 음소 상태)별로 정의되며, 클래스 수준의 난이도를 제어한다.
2. **Instance Parameters ($\sigma^{inst}$):** 개별 발화(Utterance)별로 정의되며, 샘플 수준의 난이도(예: 깨끗한 음성 vs 소음 음성)를 제어한다.

최종적으로 사용되는 데이터 파라미터 $\sigma^*_{t,i}$는 위 두 파라미터의 합으로 정의된다.
$$\sigma^*_{t,i} = \sigma^{class}_{y^i_t} + \sigma^{inst}_i$$

### 확률 계산 및 손실 함수

데이터 파라미터는 Softmax 함수에 들어가기 전 Logits을 스케일링하는 데 사용된다. 타겟 클래스 $y^i_t$에 대한 확률 $p^i_{t,y^i_t}$는 다음과 같이 계산된다.
$$p^i_{t,y^i_t} = \frac{\exp(z^i_{t,y^i_t} / \sigma^*_{t,i})}{\sum_{j} \exp(z^i_{t,j} / \sigma^*_{t,i})}$$

학습 목표는 전체 시간 프레임과 발화에 대한 평균 Cross Entropy Loss $L$을 최소화하는 것이다.
$$L = -\frac{1}{T^*} \sum_{t,i} \log(p^i_{t,y^i_t})$$
이때 $\theta$뿐만 아니라 $\sigma^{class}$와 $\sigma^{inst}$ 또한 함께 최적화한다.

### 학습 절차 및 메커니즘

데이터 파라미터의 최적화 메커니즘은 다음과 같다.

- **Gradient Scaling:** Logits에 대한 손실 함수의 기울기는 $\sigma^*_{t,i}$에 의해 반비례하게 스케일링된다. 즉, $\sigma^*$ 값이 커질수록 해당 데이터 포인트가 모델 파라미터 $\theta$ 업데이트에 미치는 영향력이 줄어든다.
- **Automatic Curriculum:** 모델이 특정 데이터를 잘못 분류하면 $\sigma^*$ 값이 점진적으로 증가하여 해당 샘플의 Gradient를 감쇠시킨다. 반대로 모델이 잘 분류하면 $\sigma^*$가 감소하여 최적화를 가속화한다. 결과적으로 모델은 초기에 쉬운 데이터를 먼저 학습하고, 어려운 데이터는 나중에 학습하게 된다.

- **최적화 세부 사항:** $\sigma$ 값이 음수가 되는 것을 방지하기 위해 $\log(\sigma)$를 최적화하며, 파라미터가 너무 극단적인 값을 갖지 않도록 $l_2$ Regularization($\|\log(\sigma^{class} + \sigma^{inst})\|^2$)을 적용하고 Clipping을 통해 범위를 제한한다.

## 📊 Results

### 실험 설정

- **데이터셋:** 약 50만 개의 깨끗한 영어 발화와 100만 개의 소음 섞인 발화(household noise, BBC sound effect 등 사용, SNR -10dB ~ 10dB)를 사용하였다.
- **모델 구조:** 64개 유닛을 가진 5층 Fully-connected Network를 사용하였으며, 입력으로는 13차원 MFCC(전후 9프레임 컨텍스트 포함, 총 247차원)를 사용하였다. 추론 단계에서는 DNN-HMM 구조를 통해 KWS 스코어를 산출한다.
- **평가 지표:** 시간당 10회의 오경보(10 FA per hour)가 발생하는 동작 지점에서 False Reject Ratio(FRR, 오거부율)를 측정하였다.

### 주요 결과

1. **소음 데이터 학습 결과 (Table 3):**
    - 단순히 소음 데이터만으로 학습한 Baseline 대비, Class와 Instance 파라미터를 모두 사용한 **Joint 설정에서 FRR이 상대적으로 7.7% 감소**하는 가장 높은 성능 향상을 보였다.
    - Class 파라미터만 사용했을 때(7.2% 향상)와 Instance 파라미터만 사용했을 때(4.1% 향상)보다 Joint 설정이 더 우수하였다.

2. **깨끗한 데이터 학습 결과 (Table 2):**
    - 깨끗한 데이터만 사용했을 때는 Class 파라미터(3.8% 향상)는 도움이 되었으나, Instance 파라미터를 적용하면 오히려 성능이 저하되는 경향을 보였다. 이는 깨끗한 데이터셋은 샘플 간 난이도 편차가 적어 Instance 파라미터가 불필요한 자유도를 제공함으로써 오버피팅을 유발했기 때문으로 분석된다.

3. **파라미터 분포 분석:**
    - 학습 초기, 소음 섞인 발화의 $\sigma^{inst}$ 평균값이 깨끗한 발화보다 더 높게 형성되었다가 시간이 흐르며 점차 비슷해지는 양상을 보였다. 이는 모델이 초기에 깨끗한 샘플에 집중하고 이후 소음 샘플을 학습하는 Curriculum Learning이 실제로 수행되었음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 데이터 파라미터를 통해 명시적인 난이도 레이블 없이도 음향 모델링에서 효과적인 Dynamic Curriculum Learning을 구현할 수 있음을 증명하였다. 특히 Multicondition Training 시 발생하는 데이터 간 난이도 불균형 문제를 자동화된 스케일링 메커니즘으로 해결하여 성능을 높였다는 점이 고무적이다.

분석 결과, Instance 파라미터의 효과는 데이터셋의 난이도 다양성이 확보되었을 때(즉, 소음 데이터가 섞여 있을 때)만 극대화된다는 점을 알 수 있다. 이는 Curriculum Learning이 모든 상황에서 유효한 것이 아니라, 학습 데이터 내에 뚜렷한 '난이도 계층'이 존재할 때 유의미함을 시사한다.

한계점으로는, 데이터 파라미터의 최적화를 위해 추가적인 하이퍼파라미터(Learning rate, Weight decay, Clipping range 등)를 튜닝해야 하며, $\sigma$ 값의 분포가 모델의 수렴 속도와 안정성에 어떤 정량적 영향을 미치는지에 대한 심층적인 분석이 부족하다는 점이 있다.

## 📌 TL;DR

본 연구는 KWS의 소음 강건성을 높이기 위해, 클래스와 인스턴스별로 학습 가능한 **Data Parameters**를 도입하여 Logits을 스케일링하는 Dynamic Curriculum Learning 기법을 제안하였다. 이 방식은 추가 레이블 없이도 모델이 쉬운 샘플부터 어려운 샘플 순으로 학습하게 하며, 소음이 포함된 데이터셋에서 기존 방식 대비 **FRR을 상대적으로 7.7% 개선**하였다. 이는 향후 다양한 환경의 음성 데이터가 존재하는 실용적인 음성 인식 시스템의 학습 효율과 성능을 높이는 데 중요한 역할을 할 수 있을 것으로 보인다.
