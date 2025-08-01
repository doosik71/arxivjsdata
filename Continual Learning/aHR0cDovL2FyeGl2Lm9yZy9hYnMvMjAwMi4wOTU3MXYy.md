# Learning to Continually Learn
Shawn Beaulieu, Lapo Frati, Thomas Miconi, Joel Lehman, Kenneth O. Stanley, Jeff Clune, Nick Cheney

## 🧩 Problem to Solve
기존 머신러닝 모델, 특히 인공 신경망은 순차적으로 여러 작업을 학습할 때 이전에 학습한 지식을 급격히 잊어버리는 **재앙적 망각(Catastrophic Forgetting, CF)**이라는 고질적인 문제에 직면합니다. 이는 딥러닝이 독립항등분포(i.i.d.) 데이터셋 환경에서 성공을 거두는 주된 이유 중 하나입니다. 대부분의 기존 CF 방지 연구는 수동으로 설계된 휴리스틱에 의존하는데, 이는 확장성 및 효율성 측면에서 한계가 있습니다. 이 논문은 AI가 스스로 지속적인 학습(continual learning)을 할 수 있도록 **메타 학습(meta-learning)**을 통해 CF 문제에 대한 해결책을 학습하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **메타 학습 기반 CF 해결:** 재앙적 망각을 직접적으로 줄이도록 최적화하는 메타 학습 접근 방식인 **A Neuromodulated Meta-Learning Algorithm (ANML)**을 제안합니다. 이는 사람이 직접 CF 방지 솔루션을 설계하는 대신 AI가 CF를 줄이는 방법을 학습하도록 합니다.
*   **신경조절(Neuromodulation) 기반 선택적 활성화:** **신경조절 네트워크(Neuromodulatory Network, NM)**가 **예측 학습 네트워크(Prediction Learning Network, PLN)**의 순방향 활성화를 조건부로 게이팅하여 **선택적 활성화(selective activation)**를 가능하게 합니다. 이는 간접적으로 **선택적 가소성(selective plasticity)**을 제어하여 네트워크의 특정 부분만 학습에 참여하도록 유도합니다.
*   **최고 수준의 지속 학습 성능 달성:** 기존 최신 연구인 OML(Online-aware Meta-Learning)을 능가하는 지속 학습 성능을 보여줍니다. 최대 600개의 순차적 클래스(9,000회 이상의 SGD 업데이트)를 재앙적 망각 없이 학습합니다.
*   **메타 학습된 희소 표현 및 클래스 분리:** 명시적으로 희소성을 장려하지 않음에도 불구하고, ANML은 희소한 활성화를 자연스럽게 생성하며, NM 네트워크는 입력 이미지 유형을 인식하고 이를 기반으로 클래스를 더 잘 분리하는 표현을 형성하여 망각을 줄입니다.

## 📎 Related Works
*   **재앙적 망각 완화 기법:**
    *   **리플레이(Replay) 방법:** 이전 경험을 저장하고 새 데이터와 혼합하여 망각을 완화하지만, 저장 및 계산 비용이 많이 들고 확장성이 떨어집니다. (예: [37])
    *   **선택적 가소성(Selective Plasticity):** 새로운 데이터에 대한 파라미터 변경 범위를 제한하여 망각을 줄입니다.
        *   **Elastic Weight Consolidation (EWC)** [19]: Fisher 정보를 통해 파라미터 중요도를 평가하여 학습 가소성을 조절합니다.
        *   **PathNet** [8], **Progressive Neural Networks** [36]: 특정 모듈을 동결하거나 새로운 용량을 추가합니다.
        *   **L2 정규화** [24], **Contrastive Excitation Backpropagation** [20, 46] 등.
    *   **희소 표현(Sparse Representations):** 활성화 간의 간섭을 최소화하기 위해 희소하거나 독립적인 표현을 생성하도록 장려합니다. (예: [11, 26])
*   **메타 학습(Meta-Learning):**
    *   **Model-Agnostic Meta-Learning (MAML)** [9]: SGD의 여러 단계를 미분하여 새로운 작업을 빠르게 학습할 수 있는 초기 가중치를 탐색합니다.
    *   **Online-aware Meta-Learning (OML)** [17]: MAML 스타일의 메타 학습을 사용하여 재앙적 망각을 최소화하는 표현(신경망 레이어 집합)을 생성하는 기존 최고 성능 모델입니다.
*   **신경조절(Neuromodulation) 기법:**
    *   신경계의 생물학적 신경조절 과정에서 영감을 받아, 학습률을 직접 조절하거나 [7, 16, 39, 40, 42] 활성화 자체를 억제하는 [32] 기존 연구들이 있습니다.

## 🛠️ Methodology
ANML은 **신경조절 네트워크 (NM Network)**와 **예측 네트워크 (Prediction Network, PLN)**라는 두 개의 병렬 신경망으로 구성됩니다 (Fig. 1 참조).

1.  **ANML 아키텍처 (ANML Architecture):**
    *   NM 네트워크와 PLN 모두 3개의 합성곱(convolutional) 레이어(각각 Batchnorm 레이어 포함)와 1개의 완전 연결(fully connected) 레이어로 구성됩니다.
    *   NM 네트워크의 최종 레이어 출력은 PLN의 최종 합성곱 레이어 출력(평탄화된 잠재 표현)과 동일한 크기를 가집니다.
    *   NM 네트워크의 출력은 PLN의 잠재 표현에 **요소별 곱셈(element-wise multiplication)**을 통해 게이팅됩니다. 이때 게이팅 값은 **시그모이드(sigmoid)** 함수를 통해 $[0,1]$ 범위로 제한되어 활성화를 억제하는 역할만 합니다.
    *   두 네트워크의 초기 가중치는 **외부 루프(outer-loop)**에서 메타 학습됩니다.

2.  **메타 학습 절차 (Meta-Training Learning Procedure) - Algorithm 1:**
    *   각 **외부 루프** 반복에서, 단일 옴니글롯(Omniglot) 메타-훈련 클래스 $T_n$의 훈련 세트(20개 인스턴스)에 대해 PLN 가중치 $\theta_P$의 복사본 $\theta_{P_0}$가 20회 **SGD 업데이트**를 통해 훈련됩니다 (**내부 루프**).
    *   이 20회 순방향 전달 동안, PLN의 최종 레이어 입력은 NM 가중치 $\theta_{NM}$에 의해 게이팅되어 **선택적 활성화**를 가능하게 합니다.
    *   역방향 전달 시, PLN의 게이팅은 특정 가중치로 흐르는 기울기(gradients)를 자연스럽게 감소시켜 **선택적 가소성**을 유도합니다. NM 가중치 $\theta_{NM}$는 내부 루프에서 업데이트되지 않습니다.
    *   20회 순차적 업데이트 후, 훈련된 클래스 20개 이미지와 **기억 세트(remember set)**(모든 메타-훈련 클래스에서 무작위 샘플링된 64개 문자 인스턴스)에 대한 예측을 사용하여 **메타 손실(meta-loss)**이 계산됩니다. 이는 OML 목적 함수를 따릅니다.
    *   이 메타 손실은 20단계의 SGD 업데이트를 통해 역전파되어 초기 PLN 가중치 $\theta_P$와 NM 네트워크 가중치 $\theta_{NM}$의 기울기를 계산하고, **Adam 옵티마이저**를 사용하여 **외부 루프 업데이트**를 수행합니다. 이 과정은 20,000회 반복됩니다.
    *   OML과 마찬가지로, 각 메타-반복 시작 시 PLN의 최종 레이어 중 다가올 클래스에 해당하는 출력 노드 가중치는 무작위로 초기화됩니다.

3.  **메타-테스트 평가 프로토콜 (Meta-Testing Evaluation Protocol) - Algorithm 2:**
    *   메타-훈련 완료 후, 학습된 PLN 초기 가중치 $\theta_P$와 NM 네트워크 $\theta_{NM}$는 재앙적 망각을 최소화하면서 많은 작업을 학습하는 능력에 대해 평가됩니다.
    *   PLN의 완전 연결 레이어만 메타-테스트 클래스에 대해 미세 조정됩니다 (각 클래스당 $q=15$ 인스턴스). 이는 OML에서 영감을 받은 부분입니다.
    *   메타-훈련과 달리, 메타-테스트 훈련에서는 각 새 클래스에 대해 PLN의 새 복사본이 만들어지지 않고, 이전 클래스에서 학습된 가중치를 이어서 미세 조정합니다 (총 9,000회 SGD 업데이트).
    *   평가는 **메타-테스트 훈련 성능**(망각 없이 기억하는 능력)과 **메타-테스트 테스트 성능**(새로운 인스턴스에 대한 일반화 능력)으로 나뉘어 측정됩니다.
    *   이 절차는 10, 50, ..., 600개의 옴니글롯 클래스 등 다양한 순차 길이에서 반복되어 모델의 확장성을 보여줍니다.

## 📊 Results
*   **재앙적 망각 저항성:**
    *   ANML은 OML, 무작위 초기화(**Scratch**), 사전 훈련 후 전이(**Pretrain & Transfer**) 등 모든 비교 대상보다 모든 길이의 메타-테스트 훈련 궤적에서 훨씬 높은 분류 정확도를 달성했습니다 (모든 $p \le 1.26 \times 10^{-8}$).
    *   특히, 600개 클래스에 대한 메타-테스트 훈련 궤적 완료 후, ANML은 63.8%의 메타-테스트 **테스트 정확도**(held-out 인스턴스에 대한 일반화 능력)를 달성하여 OML의 18.2%와 OML-OLFT(One-Layer Fine-Tuned)의 44.2%를 크게 능가했습니다 (모든 $p \le 2.58 \times 10^{-12}$).
*   **오라클(i.i.d.) 성능과의 비교:**
    *   데이터가 순차적으로 아닌 i.i.d. 방식으로 제공되는 '오라클' 버전과 비교했을 때, ANML의 성능 저하(10%에 불과)는 다른 알고리즘(OML 70.32%, Scratch 99%)보다 현저히 낮았습니다. 이는 ANML이 재앙적 망각 문제를 크게 해결했음을 시사합니다 (모든 $p \le 3.11 \times 10^{-24}$).
    *   놀랍게도, 순차적으로 훈련된 ANML은 600개 클래스에서 다른 모든 알고리즘의 **i.i.d. 훈련된 오라클 버전**(예: OML-Oracle)보다도 뛰어난 성능을 보였습니다 (모든 $p \le 1.93 \times 10^{-23}$). 이는 ANML 접근 방식이 지속 학습 설정 외에 일반적인 i.i.d. 훈련 작업에서도 성능 향상을 가져올 수 있음을 암시합니다.
*   **메타 학습된 표현 분석:**
    *   NM 네트워크의 게이팅 적용 후 PLN의 활성화된 뉴런 비율은 평균 5.9%로, 게이팅 이전의 52.77%에 비해 크게 감소했습니다 ($p < 10^{-6}$). 이는 ANML이 명시적인 희소성 장려 없이도 **희소한 표현**을 학습했음을 보여줍니다.
    *   NM 네트워크 출력의 KNN 분류 정확도는 70.9%로, 무작위 가중치를 가진 네트워크의 24.3%보다 훨씬 높습니다 ($p = 2.58 \times 10^{-31}$). 이는 NM 네트워크가 이미지 유형을 분류하는 능력을 메타 학습했음을 확인시켜 줍니다.
    *   NM 게이팅 후 PLN 활성화에 기반한 KNN 분류 정확도는 81.1%로, 게이팅 이전의 57%보다 현저히 높았습니다 ($p = 7.87 \times 10^{-12}$). 이는 NM 네트워크가 클래스 분리도(class separability)를 개선한다는 것을 정량적으로 보여줍니다 (Fig. 6의 t-SNE 시각화로도 확인 가능).

## 🧠 Insights & Discussion
*   **선택적 활성화 및 가소성의 중요성:** ANML의 핵심 통찰은 신경조절 네트워크를 통해 **선택적 활성화**를 메타 학습하는 것입니다. 이는 순방향 전달 시 간섭을 줄이고, 간접적으로 **선택적 가소성**을 유도하여 역방향 전달 시에도 망각을 방지합니다. 기존의 신경조절 연구가 주로 학습률을 조절하는 데 초점을 맞춘 반면, ANML은 활성화 자체를 직접 조절함으로써 순방향 및 역방향 모두에서 간섭을 줄이는 데 최적화되었습니다.
*   **희소성과 지속 학습의 관계:** OML과 마찬가지로 ANML도 희소한 표현을 자연스럽게 학습했지만, ANML은 OML보다 덜 희소한 표현으로도 더 나은 성능을 달성했습니다. 이는 단순히 희소한 표현만으로는 충분하지 않으며, **희소 표현과 선택적 가소성의 효과적인 조합**이 지속 학습에 더 중요하다는 것을 시사합니다.
*   **ANML의 잠재력:** 단일 에포크 및 순차적 데이터 흐름이라는 어려운 환경에서 전통적인 딥러닝 방법론(Scratch, Pretrain & Transfer)을 능가하는 ANML의 성능은 매우 인상적입니다. 또한, i.i.d. 훈련된 다른 오라클 모델들보다도 우수한 성능을 보여, 지속 학습 외의 일반적인 i.i.d. 훈련 작업에서도 ANML 아키텍처가 강력한 잠재력을 가질 수 있음을 암시합니다.
*   **한계 및 향후 연구:**
    *   현재 Omniglot 데이터셋과 같이 비교적 단순한 작업에 대한 성능이 입증되었으므로, 강화 학습(RL)과 같은 더 복잡한 작업으로의 확장 연구가 필요합니다.
    *   PLN의 한 레이어에서만 게이팅을 적용한 단순화된 가정이 있었는데, 향후 모든 레이어 또는 더 세밀한 수준(예: 각 시냅스)에서의 신경조절 적용을 탐색할 수 있습니다. 이를 위해서는 **HyperNEAT** [41]와 같은 간접 인코딩(indirect encoding) 기법을 활용하여 파라미터 폭발 문제를 해결해야 합니다.
    *   메타-훈련 분포와 메타-테스트 분포가 다른 시나리오에서의 일반화 능력 연구가 필요합니다.
    *   **미분 가능 Hebbian 학습** [33]이나 $RL^2$ [6]과 같은 다른 메타 학습 기법과의 결합도 유망한 방향입니다.
*   **AI-Generating Algorithms (AI-GAs) 패러다임:** OML과 ANML의 성공은 '가능한 한 많은 부분을 학습하도록 한다'는 AI-GAs [5]와 같은 패러다임의 강력함을 강조하며, 인공 일반 지능(Artificial General Intelligence) 연구의 큰 비전을 지지합니다.

## 📌 TL;DR
ANML은 재앙적 망각을 극복하기 위해 **신경조절 네트워크**가 **예측 네트워크**의 활성화를 게이팅하는 방법을 **메타 학습**합니다. 이는 컨텍스트 의존적인 **선택적 활성화**와 **선택적 가소성**을 가능하게 하여, 600개 클래스의 순차 학습에서 기존 최신 기술을 능가하는 지속 학습 성능을 달성합니다.