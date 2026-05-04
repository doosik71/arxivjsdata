# Adversarial Imitation Learning with Trajectorial Augmentation and Correction

Dafni Antotsiou, Carlo Ciliberto and Tae-Kyun Kim (2021)

## 🧩 Problem to Solve

딥 모방 학습(Deep Imitation Learning, IL)은 복잡한 태스크를 수행하기 위해 다량의 전문가 시연(Expert Demonstrations) 데이터를 필요로 한다. 그러나 실제 환경에서 이러한 데이터를 수집하는 과정은 기록 장치의 한계, 로봇 도메인으로의 리타겟팅 시 발생하는 노이즈, 그리고 고가의 장비 사용 등으로 인해 매우 어렵고 비용이 많이 드는 작업이다.

컴퓨터 비전 분야에서는 데이터 증강(Data Augmentation)을 통해 데이터 부족 문제를 해결하지만, 제어(Control) 태스크의 경우 문제의 순차적 특성(Sequential nature) 때문에 단순한 무작위 왜곡을 적용하면 궤적의 성공 여부가 바뀌어 버리는 문제가 발생한다. 즉, 무작위 노이즈가 추가된 궤적은 더 이상 전문가의 성공적인 궤적이 아니게 되므로, 이를 그대로 학습 데이터로 사용할 수 없다.

본 논문의 목표는 궤적의 성공 가능성을 보존하면서도 데이터의 다양성을 높일 수 있는 새로운 증강 방법을 제안하고, 이를 통해 제한된 전문가 데이터만으로도 효과적으로 학습할 수 있는 모방 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전문가의 궤적에 무작위 노이즈를 추가하여 왜곡시킨 뒤, 이를 다시 성공적인 궤적으로 복구하는 **교정 네트워크(Correction Network)**를 도입하는 것이다.

1.  **CAT (Corrected Augmentation for Trajectories):** 왜곡된 전문가 행동(Distorted actions)을 입력으로 받아 이를 성공적인 행동으로 교정하는 반지도 학습(Semi-supervised) 프레임워크를 제안한다.
2.  **DAugGI (Data Augmented Generative Imitation):** CAT 네트워크를 동적 전문가 생성기(Synthetic expert generator)로 활용하여, 무한에 가까운 합성 전문가 데이터를 생성하고 이를 통해 모방 학습 에이전트를 학습시키는 구조를 제안한다.
3.  **궤적 다양성 측정 지표:** 생성된 궤적 데이터셋의 다양성을 정량적으로 측정하기 위해 Dynamic Time Warping (DTW) 기반의 새로운 지표를 도입하여, 합성 데이터가 실제 전문가 데이터의 다양성을 유지하는지 검증한다.

## 📎 Related Works

**1. 모방 학습의 일반화 문제**
Behavioral Cloning (BC)은 학습이 쉽지만, 학습 데이터에 없는 상태에 진입했을 때 오차가 누적되는 Compounding errors 문제로 인해 일반화 능력이 떨어진다. GAIL (Generative Adversarial Imitation Learning)은 적대적 학습을 통해 이를 해결하려 했으나, 여전히 다양하고 많은 양의 전문가 데이터가 필요하다는 한계가 있다.

**2. Few-shot 및 반지도 학습**
적은 수의 전문가 데이터를 활용하려는 Meta-learning 기반의 One-shot IL 연구들이 존재한다. 또한, 노이즈가 섞인 레이블을 교정하는 연구들이 있었으나, 본 논문과 같이 제어 태스크에서 궤적을 '성공' 상태로 복구하는 관점의 접근은 차별적이다.

**3. 데이터 증강 (Data Augmentation)**
자율주행 등 제어 분야에서 이미지 증강이나 특정 행동 에뮬레이션 기반의 증강이 시도되었다. 그러나 기존 방식들은 대부분 왜곡의 결과를 미리 알고 있거나, 교정 과정에 전문가가 직접 개입해야 했다. 반면, 본 논문은 전문가의 개입 없이 적대적 학습(Adversarial method)만을 통해 전문가 분포를 맞춤으로써 데이터를 증강한다.

## 🛠️ Methodology

본 시스템은 크게 두 단계(Stage 1, Stage 2)의 파이프라인으로 구성된다.

### Stage 1: Corrected Augmentation for Trajectories (CAT)
CAT의 목적은 무작위로 왜곡된 전문가 궤적을 다시 성공적인 궤적으로 교정하는 것이다.

**1. 데이터 왜곡 과정**
전문가 궤적 $\tau^E$의 행동 $a^E$에 균일 노이즈 $\nu$ (표준편차 $\sigma$)를 더해 왜곡된 행동 $a'$를 생성한다.
$$a'_t = a^E_t + \nu$$

**2. 교정 네트워크 $\pi_\phi$**
교정 네트워크는 상태 $s$와 왜곡된 행동 $a'$를 입력으로 받아 교정된 행동 $a$를 출력한다. 이때 네트워크는 단순히 전문가를 흉내 내는 것이 아니라, 왜곡된 행동 $a'$와의 거리 $\|a - a'\|$를 최소화하면서 동시에 판별기(Discriminator)를 속여 성공적인 궤적처럼 보이게 해야 한다.

**3. 손실 함수 (Loss Function)**
교정 생성기의 손실 함수 $\mathcal{L}_\phi$는 다음과 같이 정의된다.
$$\mathcal{L}_\phi = \mathbb{E}_{\pi_\phi}[\log(1 - D_u(s, a))] + \lambda \|a - a'\|^2_2$$
여기서 첫 번째 항은 GAIL의 생성기 손실과 동일하게 판별기 $D_u$를 속이기 위한 것이며, 두 번째 항은 교정된 행동이 원래의 왜곡된 행동 $a'$에서 너무 멀어지지 않도록 가이드하는 Regularization 항이다.

### Stage 2: Data Augmented Generative Imitation (DAugGI)
Stage 1에서 학습된 CAT 네트워크를 고정(Freeze)하고, 이를 통해 합성 전문가 데이터를 동적으로 생성하여 모방 학습 에이전트 $\pi_\theta$를 학습시킨다.

**1. 동적 전문가 생성 및 필터링**
CAT는 무한한 수의 교정된 궤적을 생성할 수 있다. 다만, 모든 교정 궤적이 성공하는 것은 아니므로 **이진 성공 필터(Binary success filter)**를 적용하여 성공한 궤적만을 학습에 사용한다.

**2. 적대적 학습 구조**
DAugGI의 판별기 $D_w$는 실제 전문가가 아닌, CAT가 생성한 궤적 $\pi_\phi$와 DAugGI가 생성한 궤적 $\pi_\theta$를 구분하도록 학습된다.
$$\mathcal{L}_w = -\mathbb{E}_{\pi_\phi}[\log D_w(s, a)] - \mathbb{E}_{\pi_\theta}[\log(1 - D_w(s, a))]$$
DAugGI 생성기 $\pi_\theta$는 표준 GAIL의 생성기 손실 함수를 사용하여 $D_w$를 속이도록 학습된다.

## 📊 Results

### 실험 설정
- **태스크:** OpenAI Gym (InvertedPendulum, HalfCheetah), Dexterous Object Manipulation (Door, Hammer, Pen).
- **제한 사항:** OpenAI 태스크의 경우 전문가 데이터를 3개로 대폭 줄여 극한의 환경을 조성하였다.
- **비교 대상:** GAIL (원본 데이터 사용), DDPG (성공 여부만 알려주는 희소한 보상-Sparse binary reward- 환경에서 학습).

### 주요 결과
**1. CAT의 교정 능력**
표 I에 따르면, CAT는 무작위 왜곡(Random Augmentation)보다 훨씬 높은 성공률을 보였다. 특히 OpenAI 태스크에서는 거의 모든 궤적을 성공적으로 교정해냈으며, 복잡한 조작 태스크에서도 무작위 왜곡 대비 우수한 성능을 보였다.

**2. 모방 학습 성능 및 안정성**
- **성공률:** DAugGI는 특히 중간 난이도 태스크(HalfCheetah, Door)에서 GAIL 대비 뚜렷한 성능 향상을 보였다.
- **안정성:** GAIL은 수렴 과정에서 매우 불안정한 모습을 보인 반면, DAugGI는 훨씬 안정적인 수렴 특성을 나타냈다.
- **희소 보상 RL과의 비교:** DDPG는 성공 필터만으로는 학습이 불가능(Convergence 실패)했으나, DAugGI는 이를 활용해 성공적으로 학습하였다.

**3. 다양성 분석 (Diversity)**
DTW 기반의 다양성 지표 $\text{dtw}_n$을 통해 측정했다.
$$\text{dtw}_n(T^g) = \frac{\text{dtw}(T^g)}{\text{dtw}(T^E)}$$
분석 결과, DAugGI로 학습된 에이전트의 궤적 다양성은 GAIL과 유사하거나 오히려 약간 더 높게 나타났다. 이는 CAT를 통한 데이터 증강이 전문가 데이터셋이 가진 상태-행동 공간의 표현력을 확장시켰음을 시사한다.

## 🧠 Insights & Discussion

**강점 및 해석**
본 연구는 제어 시스템에서 데이터 증강 시 발생하는 '레이블 변동(성공 $\rightarrow$ 실패)' 문제를 교정 네트워크라는 아이디어로 해결하였다. 특히, CAT 네트워크가 왜곡된 행동 $a'$라는 가이드를 받기 때문에 일반적인 모방 학습보다 학습 속도가 빠르고 안정적이며, 이렇게 생성된 '준전문가' 데이터가 최종 에이전트(DAugGI)에게 훌륭한 징검다리 역할을 하여 학습 성능을 높인 것으로 분석된다.

**한계 및 논의사항**
- **태스크 난이도 의존성:** Pen 조작과 같이 매우 어려운 태스크에서는 CAT의 교정 능력이 떨어져 DAugGI의 성능 향상이 미미했다. 이는 교정 네트워크 자체가 충분히 좋은 '교사'가 되지 못할 경우 효과가 제한적임을 의미한다.
- **가정:** 본 모델은 각 태스크의 성공 여부를 판단할 수 있는 이진 필터(Binary filter)가 존재한다고 가정한다. 실제 환경에서 이러한 성공 기준을 명확히 정의하는 것이 어려울 수 있다.

## 📌 TL;DR

본 논문은 전문가 데이터 부족 문제를 해결하기 위해, 왜곡된 궤적을 성공적인 궤적으로 복구하는 **CAT(교정 네트워크)**와 이를 활용해 동적으로 데이터를 증강하는 **DAugGI(모방 학습 네트워크)** 프레임워크를 제안한다. 실험 결과, 제안 방법은 적은 양의 전문가 데이터만으로도 기존 GAIL보다 높은 성공률과 학습 안정성을 보였으며, 데이터의 다양성 또한 유지하거나 개선하였다. 이 연구는 데이터 수집 비용이 높은 로봇 제어 및 복잡한 조작 태스크의 학습 효율을 높이는 데 기여할 가능성이 크다.