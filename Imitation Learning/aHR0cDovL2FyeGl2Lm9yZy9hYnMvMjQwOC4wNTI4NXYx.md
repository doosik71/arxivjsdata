# Semi-Supervised One-Shot Imitation Learning

Philipp Wu, Kourosh Hakhamaneshi, Yuqing Du, Igor Mordatch, Aravind Rajeswaran, Pieter Abbeel (2024)

## 🧩 Problem to Solve

본 논문은 One-Shot Imitation Learning (OSIL)에서 발생하는 데이터 효율성 문제를 해결하고자 한다. OSIL의 목표는 단 한 번의 시연(demonstration)만으로 새로운 태스크를 학습하여 수행하는 능력을 AI 에이전트에게 부여하는 것이다. 

기존의 OSIL 방식은 학습을 위해 방대한 양의 '쌍을 이룬 전문가 시연(paired expert demonstrations)' 데이터셋을 필요로 한다. 여기서 쌍을 이룬 데이터란 동일한 시맨틱 태스크(semantic task)를 수행하지만 서로 다른 환경 변형을 가진 궤적(trajectory)들의 집합을 의미한다. 이러한 데이터를 대량으로 수집하는 것은 상당한 엔지니어링 비용과 인간의 주석(annotation) 시간이 소요되므로 실질적인 적용에 큰 제약이 된다.

따라서 본 연구의 목표는 레이블이 없는 대규모의 궤적 데이터셋(unpaired dataset)과 레이블이 있는 소규모의 쌍을 이룬 데이터셋(paired dataset)을 동시에 활용하는 '준지도 OSIL(Semi-supervised OSIL)' 설정과 이를 위한 알고리즘을 제안하여 레이블 효율성을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"태스크를 완수하기 위한 액션을 생성하는 모델을 학습시키는 것보다, 동일한 시맨틱 태스크를 공유하는 궤적들을 구분하거나 클러스터링하는 것이 더 쉽고 데이터 효율적이다"**라는 직관에 기반한다.

이를 위해 저자들은 **Student-Teacher trajectory relabeling** 접근 방식을 제안한다. 소량의 레이블된 데이터로 먼저 Teacher Encoder를 학습시켜 임베딩 공간을 구축하고, 이 공간에서의 근접 이웃(k-Nearest Neighbors, kNN) 검색을 통해 레이블이 없는 데이터에 대해 의사 레이블(pseudo-labels)을 생성한다. 이렇게 생성된 의사 쌍(pseudo-pairs)을 활용해 Student Policy를 학습시킴으로써, 매우 적은 양의 정답 레이블만으로도 완전 지도 학습(fully supervised)에 근접한 성능을 달성한다.

## 📎 Related Works

**1. One-Shot Imitation Learning (OSIL)**
Duan et al. (2017)에 의해 도입된 OSIL은 메타 학습(meta-learning) 관점에서 접근하며, 한 궤적을 조건으로 다른 쌍을 이룬 궤적을 재구성함으로써 태스크의 시맨틱을 암시적으로 학습한다. 이후 시각적 관측 공간 확장이나 트랜스포머 구조 도입 등의 연구가 있었으나, 여전히 대량의 쌍을 이룬 데이터가 필요하다는 한계가 있다.

**2. Semi-Supervised Learning (SSL)**
컴퓨터 비전과 NLP 분야에서 널리 쓰이는 SSL은 소량의 레이블 데이터와 대량의 무레이블 데이터를 함께 사용한다. 특히 본 논문에서 채택한 **Self-training** 방식은 Teacher 모델이 무레이블 데이터에 의사 레이블을 부여하고, Student 모델이 이를 학습하는 구조를 가진다.

**3. Semi-Supervised Learning in RL and IL**
기존의 RL/IL 분야에서는 보상(reward)이나 목표(goal) 레이블이 없는 경우 역강화학습(Inverse RL)이나 적대적 모방 학습(Adversarial IL) 등을 통해 해결하려 했다. 하지만 OSIL의 레이블 효율성을 높이기 위해 준지도 학습 접근법을 적용한 연구는 본 논문이 처음으로 시도한 것으로 명시되어 있다.

## 🛠️ Methodology

### 전체 시스템 구조
OSIL 에이전트는 크게 두 가지 모듈로 구성된다.
1. **Demonstration Encoder ($f_\phi(d)$):** 시연 궤적 $d$를 잠재 공간(latent space)의 임베딩 $z$로 변환한다.
2. **Policy Decoder ($\pi_\theta(a_t|s_t, z)$):** 현재 상태 $s_t$와 궤적 임베딩 $z$를 입력받아 액션 $a_t$를 출력한다.

### 학습 절차 (Student-Teacher Training)

**단계 1: Teacher Encoder 학습**
소규모의 레이블된 데이터셋 $D_{labeled}$를 사용하여 Teacher Encoder와 Policy를 End-to-End로 학습시킨다. 이때 두 가지 손실 함수를 사용한다.
- **Imitation Loss:** 정책이 예측한 액션과 전문가의 액션 간의 오차를 최소화한다.
- **Contrastive InfoNCE Loss:** 동일 태스크의 궤적들은 임베딩 공간에서 가깝게, 서로 다른 태스크는 멀게 배치하여 구조화된 잠재 공간을 형성하도록 유도한다. 

**단계 2: 의사 레이블 생성 (Pseudo-labeling)**
학습된 Teacher Encoder $f_\phi$를 사용하여 무레이블 데이터셋 $D_{unlabeled}$의 모든 궤적을 임베딩한다. 각 궤적 $d_i$에 대해 임베딩 공간에서 $L_2$ 거리가 가장 가까운 $k$개의 이웃을 찾아 의사 쌍(pseudo-pairs)을 생성한다.
$$D_{pseudo\_labeled} = \{(d_i, \{kNN_\phi(d_i, D_{unlabeled})\}) \forall d_i \in D_{unlabeled}\}$$

**단계 3: Student Policy 학습**
Student 모델은 $D_{labeled}$와 $D_{pseudo\_labeled}$를 모두 사용하여 학습한다. 
- $D_{labeled}$에 대해서는 Imitation Loss와 Contrastive Loss를 모두 적용한다.
- $D_{pseudo\_labeled}$에 대해서는 오직 Imitation Loss만을 적용하여 학습한다.

이 과정은 Student 모델의 Encoder를 다시 Teacher로 사용하여 반복적으로 수행할 수 있다.

### 상세 아키텍처
- **Policy:** MLP 구조를 사용하며, 임베딩 $z$를 입력하기 위해 FiLM(Feature-wise Linear Modulation) 컨디셔닝 기법을 적용한다.
- **Image Encoder:** 64x64 이미지를 처리하기 위해 5계층 CNN을 사용한다.
- **Trajectory Encoder:** 
    - 단순한 태스크(예: 목표 지점 도달)의 경우, 마지막 프레임의 임베딩만을 사용한다.
    - 복잡한 순차적 태스크(Sequential Navigation)의 경우, 각 프레임을 토큰으로 취급하여 처리하는 **Bi-directional Transformer**를 사용하여 궤적 전체의 시간적 맥락을 인코딩한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업:**
    - **Semantic Goal Navigation:** 다양한 색상과 모양의 목표물 중 하나를 찾아가는 태스크.
    - **Sequential Goal Navigation:** 6개의 버튼 중 2개를 정해진 순서대로 누르는 태스크 (시간적 순서 중요).
- **지표:** 
    - **Task Success Rate:** 100회 시도 중 성공 횟수의 평균.
    - **Trajectory Retrieval (TR) Score:** 임베딩 공간에서 $k$개의 근접 이웃을 찾았을 때, 이들이 실제로 동일한 태스크인지 측정하는 정확도.

### 주요 결과
- **Semantic Goal Navigation:** 레이블된 데이터의 15%만 사용했음에도 불구하고, 제안 방법(Imit+Cont+Aug+Relabel)은 100% 레이블을 사용한 모델과 대등한 성능을 보였다.
- **Sequential Goal Navigation:** 더 복잡한 이 작업에서도 단 5%의 레이블된 데이터만으로 완전 지도 학습 모델의 성능에 근접하는 결과를 얻었다.
- **TR Score 분석:** 데이터 양이 줄어들어도 Trajectory Retrieval 점수는 높게 유지되었다. 이는 정책 학습보다 임베딩 공간의 클러스터링 학습이 훨씬 데이터 효율적임을 시사한다.
- **Ablation Study:** Contrastive Loss는 임베딩 공간의 구조화(Retrieval 성능 향상)에 필수적이며, Relabeling 과정은 실제 정책의 성공률을 높이는 데 결정적인 역할을 함을 확인하였다.

## 🧠 Insights & Discussion

**강점 및 통찰**
본 논문은 OSIL에서 가장 비용이 많이 드는 '쌍을 이룬 데이터'의 수집 문제를 준지도 학습의 Self-training 메커니즘으로 해결하였다. 특히, 생성적 모델링(액션 예측)보다 판별적 모델링(궤적 클러스터링)이 훨씬 쉽다는 점을 이용하여, 상대적으로 쉬운 태스크인 '유사 궤적 찾기'를 통해 어려운 태스크인 '정책 학습'을 가속화한 점이 돋보인다.

**한계 및 논의사항**
1. **태스크 복잡도에 따른 영향:** 저자들은 단순한 태스크에서는 $k$값의 선택이나 반복적 Relabeling의 횟수가 성능에 큰 영향을 주지 않았다고 언급하였다. 이는 더 복잡하고 정교한 제어가 필요한 태스크에서는 이러한 하이퍼파라미터가 훨씬 더 중요한 변수가 될 가능성이 크다.
2. **의사 레이블의 품질:** Teacher 모델이 생성한 의사 레이블에 오류가 있을 경우, Student 모델이 이를 학습하며 성능이 저하될 위험(error propagation)이 존재한다. 다만, 본 실험에서는 Contrastive Loss와 대량의 무레이블 데이터를 통해 이 문제를 완화한 것으로 보인다.

## 📌 TL;DR

본 논문은 대량의 레이블 없는 궤적 데이터와 소량의 레이블된 데이터를 활용하는 **준지도 One-Shot Imitation Learning (Semi-supervised OSIL)** 프레임워크를 제안한다. **Teacher-Student 구조**를 통해 임베딩 공간에서 유사한 궤적들을 찾아 의사 레이블(pseudo-labels)을 생성하고, 이를 통해 정책을 학습시킨다. 결과적으로 **전체 데이터의 5~15%만 레이블링되어 있어도 완전 지도 학습과 유사한 성능**을 낼 수 있음을 증명하여, OSIL의 실용적인 데이터 효율성을 획기적으로 높였다.