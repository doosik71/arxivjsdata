# ACTIVEWEASUL: IMPROVING WEAK SUPERVISION WITH ACTIVE LEARNING

Samantha Biegel, Rafah El-Khatib, Luiz Otavio Vilas Boas Oliveira, Max Baak & Nanne Aben (2021)

## 🧩 Problem to Solve

머신러닝 모델의 성능을 높이기 위해서는 대량의 레이블링된 데이터가 필요하지만, 실제 환경에서 고품질의 레이블을 확보하는 것은 매우 어렵다. 특히 도메인 전문가의 지식이 필요한 복잡한 작업이나 개인정보 보호 문제로 인해 데이터를 외부로 공유할 수 없는 경우, 레이블링 비용은 기하급수적으로 증가한다.

이를 해결하기 위해 전문가가 정의한 규칙(Rule)을 통해 확률적 레이블을 추정하는 Weak Supervision(약지도 학습) 프레임워크가 제안되었다. 그러나 Weak Supervision은 전문가가 정의한 규칙의 정확도에 의존하므로, 규칙 자체가 부정확하거나 문제 공간의 일부만을 포착하는 경우 모델의 성능이 최적 수준에 도달하지 못한다는 한계가 있다.

본 논문의 목표는 Active Learning(능동 학습)을 Weak Supervision에 통합하여, 아주 적은 양의 전문가 레이블링만으로도 Weak Supervision 모델의 성능을 효율적으로 개선하는 **Active WeaSuL** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Weak Supervision을 통해 모델에 'Warm Start(초기 가속)'를 제공하고, 이후 Active Learning을 통해 모델이 가장 불확실해하는 지점을 집중적으로 보완하는 것이다. 이를 위해 다음과 같은 두 가지 핵심 기여를 수행한다.

1.  **Weak Supervision 손실 함수 수정**: 전문가가 직접 레이블링한 소량의 데이터가 Weak Supervision의 생성 모델(Generative Model)에 반영되어, 약한 레이블(Weak Labels)들을 더 정확하게 결합할 수 있도록 하는 패널티 항을 도입한다.
2.  **maxKL Divergence 샘플링 전략**: 생성 모델의 예측값과 실제 전문가 레이블 간의 차이가 가장 큰 데이터 지점을 선택하는 `maxKL` 샘플링 전략을 제안하여, 전문가의 레이블링 효율을 극대화한다.

## 📎 Related Works

기존 연구 중 Active Learning과 Weak Supervision을 결합한 대표적인 사례로 Nashaat et al. (2019)의 연구가 있다. 그러나 두 접근 방식은 전문가 데이터를 활용하는 방식에서 근본적인 차이가 있다.

- **Nashaat et al. (2019)**: 전문가가 레이블링한 데이터를 사용하여 해당 개별 데이터 포인트의 약한 레이블을 직접 수정한다. 이는 특정 데이터에 대한 정답은 맞출 수 있게 하지만, 모델 전체의 레이블 결합 방식을 개선하지는 못하는 국소적인(Local) 해결책이다.
- **Active WeaSuL**: 전문가 데이터를 사용하여 약한 레이블들을 어떻게 결합하는 것이 최선인지를 학습한다. 이는 모델 전체의 매개변수를 조정하여 모든 데이터 포인트의 예측 성능을 높이는 전역적인(Global) 해결책이다.

또한, 일부 연구들은 레이블링 함수(Labelling Function) 자체를 자동으로 개선하려 하거나, 반대로 Weak Supervision을 통해 Active Learning의 초기 단계를 돕는 방식을 취하지만, 본 논문은 고정된 규칙 하에서 레이블 결합 최적화에 집중한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Weak Supervision 기초
본 논문은 Ratner et al. (2019)의 생성 모델을 기반으로 한다. $n$개의 샘플과 $m$개의 약한 레이블이 있을 때, 약한 레이블 행렬 $\lambda$의 공분산 행렬 $\Sigma$와 조건부 독립 구조 $\Omega$를 이용하여 다음과 같은 목적 함수를 최적화하여 파라미터 $\hat{z}$를 찾는다.

$$\hat{z} = \text{argmin}_z \|(\Sigma^{-1} + zz^T)_\Omega\|_F$$

여기서 $\|.\|_F$는 Frobenius norm이며, 이렇게 구해진 $\hat{z}$를 통해 각 데이터 포인트의 확률적 레이블 $\hat{p}(y|\lambda_{i,*}) = f(\hat{z}, \lambda_{i,*})$를 도출한다.

### 2. 수정된 생성 모델 손실 함수
Active Learning을 통해 얻은 전문가 레이블 데이터 셋 $D$를 반영하기 위해, 기존 손실 함수에 패널티 항 $P_e(z)$를 추가한다.

$$\hat{z} = \text{argmin}_z \|(\Sigma^{-1} + zz^T)_\Omega\|_F + \alpha P_e(z)$$

이때 패널티 항 $P_e(z)$는 생성 모델이 예측한 확률적 레이블과 전문가가 부여한 실제 이진 레이블 $y_i$ 사이의 이차 차이(Quadratic Difference)의 합으로 정의된다.

$$P_e(z) = \sum_{i \in D} (f(z, \lambda_{i,*}) - y_i)^2$$

하이퍼파라미터 $\alpha$는 패널티 항의 영향력을 조절하며, 이를 통해 모델은 전문가의 레이블과 일치하는 방향으로 약한 레이블들을 결합하도록 강제된다.

### 3. maxKL Divergence 샘플링 전략
전문가에게 어떤 데이터를 레이블링 요청할지 결정하기 위해 `maxKL` 전략을 사용한다.

1.  **Bucket 정의**: 약한 레이블들의 고유한 조합(Configuration)을 하나의 '버킷(Bucket)'으로 정의한다.
2.  **두 가지 확률 추정**:
    - $\hat{p}^t(y|\gamma_{i,*})$: 현재 반복 회차 $t$에서 생성 모델이 추정한 버킷 $i$의 확률.
    - $\hat{q}^t(y|\gamma_{i,*})$: 버킷 $i$ 내에서 전문가가 레이블링한 데이터만을 이용해 계산한 경험적 확률.
3.  **KL Divergence 계산**: 두 분포 $\hat{p}$와 $\hat{q}$ 사이의 KL Divergence를 계산하여 모델과 전문가의 의견 차이가 가장 큰 버킷을 찾는다.

$$KL_{t,i}(p||q) = p \cdot \log \frac{p}{q} + (1-p) \cdot \log \frac{1-p}{1-q}$$

가장 높은 KL Divergence 값을 가진 버킷에서 데이터 포인트를 샘플링하여 전문가에게 레이블링을 요청하며, 이 과정을 반복하여 모델을 업데이트한다.

## 📊 Results

### 실험 설정
- **인공 데이터셋**: 2차원 가우시안 분포 기반의 이진 분류 문제.
- **VRD (Visual Relationship Detection)**: 이미지와 객체 정보를 이용해 '대상(subject)이 물체(object) 위에 앉아 있는가'를 판별하는 작업 (F1 Score 측정).
- **스팸 탐지**: YouTube 댓글의 스팸 여부를 판별하는 작업 (F1 Score 측정).
- **비교 대상**: 순수 Weak Supervision, 순수 Active Learning, Nashaat et al. (2019)의 방식.

### 주요 결과
1.  **성능 향상 속도**: 인공 데이터셋에서 Active WeaSuL은 단 4회의 Active Learning 반복만으로 정확도를 $0.81 \rightarrow 0.96$으로 끌어올렸다. 순수 Active Learning은 동일 성능 도달까지 더 많은 반복이 필요했으며, Nashaat et al.의 방식은 1,000회 반복 후에도 Active WeaSuL의 초기 성능에 미치지 못했다.
2.  **적은 레이블 비용**: VRD 작업에서 Active WeaSuL은 단 7개의 레이블만으로 F1 스코어 0.58을 달성했다. 반면 순수 Active Learning은 66개, Nashaat et al. 방식은 77개의 레이블이 필요했다.
3.  **샘플링 전략 비교**: `maxKL` 전략이 `margin` 샘플링이나 `random` 샘플링보다 월등한 성능을 보였다. 특히 `margin` 전략은 특정 버킷에 편향되어 샘플링하는 경향(낮은 엔트로피)이 있는 반면, `maxKL`은 정보성과 다양성 사이의 균형을 잘 유지했다.

## 🧠 Insights & Discussion

본 연구는 Weak Supervision이 제공하는 'Warm Start'가 Active Learning의 초기 수렴 속도를 획기적으로 높일 수 있음을 입증했다. 전문가가 정의한 규칙이 다소 편향되어 있더라도, 소량의 정답 데이터를 통해 그 편향을 보정하는 방향으로 결합 가중치를 조정하는 것이 개별 데이터를 수정하는 것보다 훨씬 효율적이다.

**한계 및 비판적 해석**:
- **레이블 예산에 따른 선택**: 실험 결과, 레이블 예산이 매우 충분하다면 결국 순수 Active Learning(완전 지도 학습)이 Active WeaSuL을 추월한다. 이는 약지도 학습의 규칙이 어느 시점부터는 모델의 학습을 제약하는 요소로 작용하기 때문이다. 따라서 본 방법론은 '레이블 예산이 극도로 제한된 상황'에서 최적의 효용을 가진다.
- **규칙의 의존성**: 본 방법론은 여전히 초기에 전문가가 정의한 기본 규칙(Labelling Functions)이 어느 정도 유효하다는 가정 하에 작동한다. 만약 규칙이 완전히 잘못되었다면 Warm Start의 효과가 미비할 가능성이 있다.

## 📌 TL;DR

Active WeaSuL은 전문가의 **'규칙(Weak Supervision)'**과 **'소량의 정답 데이터(Active Learning)'**를 결합한 프레임워크이다. 수정된 손실 함수를 통해 약한 레이블의 결합 방식을 전역적으로 최적화하고, `maxKL` 전략으로 가장 효율적인 데이터 포인트를 선택한다. 이 연구는 레이블링 비용이 매우 높은 전문 분야에서, 아주 적은 양의 데이터만으로도 빠르게 고성능 모델을 구축할 수 있는 실무적인 경로를 제시한다.