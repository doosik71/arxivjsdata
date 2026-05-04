# A Geometric Perspective on Visual Imitation Learning

Jun Jin, Laura Petrich, Masood Dehghan, and Martin Jagersand (2020)

## 🧩 Problem to Solve

본 논문은 인간의 직접적인 감독(예: Kinesthetic teaching 또는 Teleoperation)이나 상호작용 가능한 강화학습(RL) 환경에 대한 접근 없이, 오직 시각적 관찰만으로 로봇이 작업을 학습하는 Visual Imitation Learning의 일반화(Generalization) 문제를 해결하고자 한다.

기존의 행동 복제(Behavior Cloning) 방식은 방대한 양의 데이터가 필요하며, RL 기반 방식은 시뮬레이션과 실제 환경 간의 차이(Sim-to-Real gap)와 낮은 샘플 효율성 문제가 존재한다. 또한, 많은 기존 방법론들이 이미지 픽셀에서 액션으로 직접 매핑하는 정책을 학습하려 하기 때문에, 시연자와 모방자(로봇) 사이의 외형적 차이나 환경 변화에 취약하다는 한계가 있다. 따라서 본 연구의 목표는 인간의 시연 영상에서 '무엇을 해야 하는가'에 해당하는 고수준의 작업 개념(Task Concept)을 기하학적으로 추론하고, 이를 로봇의 저수준 제어와 효율적으로 연결하는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 작업의 정의('What')와 실제 제어('How')를 분리하는 계층적 관점을 도입하고, 이를 **기하학적 파라미터화(Geometry-parameterization)**를 통해 구현하는 것이다.

1. **VGS-IL (Visual Geometric Skill Imitation Learning) 제안**: 이미지 픽셀로부터 전역적으로 일관된 기하학적 특징 연관 규칙(Geometric feature association rules)을 추론하는 엔드투엔드 방법론을 제안한다. 이는 수작업으로 설계된 특징 기술자(Feature descriptor) 대신 데이터 기반으로 기하학적 개념을 최적화한다.
2. **설명 가능하고 불변한 표현 학습**: 픽셀-액션 매핑 대신 기하학적 파라미터로 작업을 표현함으로써, 시연자와 로봇의 외형 차이가 있더라도 변하지 않는(Invariant) 표현을 얻을 수 있으며, 이는 작업의 성격을 명확히 설명할 수 있게 한다.
3. **기하학적 비전 기반 제어기와의 직접 연결**: 학습된 기하학적 작업 개념을 Visual Servoing과 같은 제어기에 직접 연결함으로써, 별도의 복잡한 저수준 정책 학습 없이도 고수준 개념을 로봇의 동작으로 효율적으로 변환할 수 있음을 보였다.

## 📎 Related Works

논문에서는 Visual Imitation Learning의 접근 방식을 크게 세 가지로 분류하여 설명한다.

- **Hierarchical Visual Imitation Learning**: 작업 정의와 제어를 분리하는 방식으로, 픽셀 수준(Sub-goal output)이나 객체 수준(Object correspondence, Graph structure)에서 작업을 표현한다. 그러나 객체 수준의 표현은 삽입(Insertion)과 같은 정밀한 작업에 필요한 해상도가 부족하다는 한계가 있다.
- **Geometry-Based Visual Imitation Learning**: 특징점(Keypoint) 간의 관계를 통해 작업을 표현하는 방식이다. 하지만 기존 연구들은 저수준 제어기를 별도로 어렵게 학습시켜야 했으며, 적절한 작업 표현이 제어기 학습에 어떤 영향을 주는지에 대한 연구가 부족했다.
- **차별점**: 본 논문은 단순히 특징점 대응을 넘어 점-점(Point-to-point), 점-선(Point-to-line) 등 다양한 기하학적 제약 조건을 체계적으로 결합하여 복잡한 작업을 정의하고, 이를 Visual Servoing 제어기에 직접 연결하여 학습 비용을 획기적으로 줄였다.

## 🛠️ Methodology

### 1. 기하학적 파라미터화 작업 표현 (Geometry parameterized task representation)

작업은 하나 이상의 **VGS Kernel**들의 조합 또는 연결로 정의된다. 주요 커널의 예시는 다음과 같다.

- $gk_{p2p}$: 두 점의 일치 (Point-to-point)
- $gk_{p2l}$: 점이 선 위에 위치 (Point-to-line)
- $gk_{l2l}$: 두 선의 공선성 (Line-to-line)
- $gk_{p2c}$: 점이 원추곡선 위에 위치 (Point-to-conic)

이러한 커널 $gk$는 다음의 세 가지 속성을 만족해야 한다.

- **Communicative**: 입력 특징의 순서가 바뀌어도 결과가 동일해야 한다.
- **Non-inner-associative**: 내부 연관 구조에 따라 서로 다른 의미를 가져야 한다 (예: 점 하나와 선을 이루는 점 세 개의 결합은 서로 다른 연산임).
- **Scalability**: $n$-ary 연산으로 확장 가능해야 한다.

이를 위해 본 논문은 **Message Passing Graph Neural Network (GNN)**와 **GRU (Gated Recurrent Unit)**를 사용하여 커널을 구현하였다. 구체적인 단계는 다음과 같다.

1. **메시지 생성**: 연결된 노드들의 은닉 상태를 통해 메시지를 생성한다.
    $$m_{i \to j}^{t+1} = M(h_i^t, h_j^t)$$
2. **메시지 집계**: 모든 들어오는 메시지를 수집한다.
    $$m_i^{t+1} = A(m_{i \to j}^{t+1})$$
3. **메시지 업데이트**: GRU를 사용하여 노드의 상태를 업데이트한다.
    $$h_i^{t+1} = U(h_i^t, m_i^{t+1})$$
4. **Readout**: 최종 상태들을 MLP에 통과시켜 잠재 벡터 $b$를 산출한다.
    $$b = \text{MLP}(h_1^T, \dots, h_n^T)$$

### 2. VGS-IL 학습 과정

인간의 시연 영상 $\{I_t\}$가 주어졌을 때, 최적의 $gk_i$를 찾는 과정은 다음과 같다.

- **Select-out 함수**: 가능한 모든 조합의 인스턴스들 중 최적의 기하학적 특징 결합을 선택한다. 각 인스턴스의 출력 $b_j$에 대해 Softmax를 적용하여 관련성 $g_j$를 계산하고, 가장 높은 값을 가진 인스턴스를 선택한다.
- **최적화 (Observational Expert Assumption)**: 최적의 $gk_i$를 적용하면 인간 시연자의 영상에서 고품질의 제어 오차 신호(Control error signal) $e_t$가 도출될 것이라는 가정을 세운다.
- **손실 함수**: 오차 신호가 시간에 따라 전반적으로 감소하는지와 신호가 부드럽게 변화하는지를 측정한다. 특히 부드러움을 위해 **Geometry Consistent Regularizer (GCR)**를 추가하였다.
    $$\text{Loss} = \dots - \alpha \|b_{t+1} - b_t\|_2^2$$
    이 최적화는 $\text{InMaxEntIRL}$ 알고리즘을 통해 수행된다.

### 3. 제어기와의 연결

VGS-IL을 통해 도출된 오차 벡터 $\dot{e}_t$는 이미지 픽셀 공간에 존재한다. 이를 로봇의 실제 동작으로 매핑하기 위해 **Visual Servoing (VS)** 기법을 사용한다.

- **매핑 과정**: 이미지 상의 오차 $\dot{e}_t \to$ 카메라의 속도 $v_c^t \to$ 로봇의 조인트 속도 $a_t$ 순으로 매핑이 이루어진다.
- 이 과정에서 Jacobian 행렬(Interaction matrix)과 캘리브레이션 모델을 사용하여 별도의 강화학습 없이도 즉각적인 제어가 가능하다.

## 📊 Results

### 실험 설정

- **대상 작업**: Sorting(점-점), Insertion(점-점 및 점-선 조합), Folding(변형 가능한 객체), Screw(저텍스처 환경)
- **비교 대상 (Baselines)**:
  - Baseline 1: 수작업으로 특징을 선택하고 비디오 트래킹을 사용하는 전통적인 Visual Servoing.
  - Baseline 2: SIFT/LBD 기술자를 사용한 이전 연구 방식 (일관성 정규화 없음).
- **평가 지표**: 정확도($Acc$: 올바른 특징 추론 프레임 비율)와 일관성($conAcc$: 오차 노름의 자기상관관계 측정).

### 주요 결과

1. **추론 성능 (Table I)**: VGS-IL은 모든 작업에서 Baseline 2보다 월등히 높은 일관성($conAcc$)을 보였으며, 정밀한 작업인 Insertion과 Screw에서도 높은 정확도를 기록하였다.
2. **환경 일반화 성능 (Table II)**: 무작위 타겟 위치, 카메라 위치 변경, 객체 가려짐(Occlusion), 시야 밖으로 벗어남(Outside FOV), 조명 변화 등 다양한 가혹 조건에서 VGS-IL이 가장 높은 성능을 유지하였다. 특히 시야 밖으로 나갔다 돌아왔을 때의 복구 능력이 뛰어났다.
3. **제어 신호 품질 (Fig. 8)**: 학습 단계($S1 \to S2 \to S3$)가 진행됨에 따라 출력되는 제어 오차 신호 $e_t$가 점점 더 매끄러워지고 일관되게 감소하는 것을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 시각적 모방 학습에서 '무엇을 학습할 것인가'에 대한 기하학적 관점을 제시함으로써 다음과 같은 통찰을 제공한다.

- **강점**: 픽셀 기반의 직접 학습 대신 기하학적 불변량(Invariant)을 학습함으로써, 시연자와 로봇의 외형적 차이를 극복하고 환경 변화에 매우 강건한 일반화 성능을 확보하였다. 또한, Visual Servoing과의 결합을 통해 복잡한 제어 정책 학습 과정을 생략할 수 있었다.
- **한계 및 미해결 과제**:
  - **계산 비용**: 모든 조합 가능한 특징 연관 후보들에 대해 최적화를 수행하므로 GPU 계산 자원 소모가 매우 크다. 저자는 이에 대한 대안으로 Bayesian Optimization의 도입을 언급한다.
  - **작업 확장성**: 본 논문에서는 커널의 '조합(Combination)'만을 다루었으며, 커널들을 '순차적으로 연결(Sequential linking)'하여 더 복잡한 다단계 작업을 수행하는 방법은 향후 연구 과제로 남겨두었다.
- **비판적 해석**: 제안된 방법은 기하학적 구조가 명확한 작업에는 매우 효율적이지만, 기하학적으로 정의하기 어려운 비정형 작업(예: 액체 젓기, 천 접기 중의 세밀한 변형 등)에 대해서는 어떻게 파라미터화할 것인지에 대한 논의가 더 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 인간의 시연 영상에서 기하학적 특징 연관 규칙을 추론하는 **VGS-IL** 방법론을 제안한다. 이 방법은 작업을 '기하학적 커널'의 조합으로 파라미터화하여 시연자와 로봇 간의 외형 차이에 영향을 받지 않는 불변한 작업 표현을 학습하며, 이를 Visual Servoing 제어기에 직접 연결해 효율적인 로봇 제어를 가능케 한다. 결과적으로 다양한 환경 변화 속에서도 높은 일반화 성능을 보였으며, 이는 향후 로봇의 범용 작업 프로그래밍에 중요한 기하학적 프레임워크를 제공한다.
