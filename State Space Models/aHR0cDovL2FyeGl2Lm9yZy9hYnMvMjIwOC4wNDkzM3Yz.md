# SIMPLIFIED STATESPACE LAYERS FOR SEQUENCE MODELING
Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman

## 🧩 Problem to Solve
이 논문은 장거리 시퀀스 모델링(long-range sequence modeling)의 효율성 문제를 다룹니다. 기존 시퀀스 모델(RNN, CNN, Transformer 등)은 시퀀스 길이가 길어질수록 효율성(계산 복잡도) 및 성능 저하 문제를 겪습니다. 특히, Gu et al. (2021a)이 제안한 S4(Structured State Space Sequence) 레이어는 최첨단 성능을 달성했지만, 다수의 독립적인 단일 입력, 단일 출력(SISO) 선형 상태 공간 모델(SSM)을 사용하며, 복잡한 컨볼루션 및 주파수 도메인 접근 방식을 통해 구현되어 계산 효율성과 구현 복잡성 측면에서 개선의 여지가 있었습니다.

## ✨ Key Contributions
*   **S5 레이어 도입:** S4 레이어의 설계를 기반으로 단일 다중 입력, 다중 출력(MIMO) SSM을 사용하는 새로운 상태 공간 레이어인 S5 레이어를 제안합니다. S4는 다수의 독립적인 SISO SSM을 사용합니다.
*   **효율적인 병렬 스캔 활용:** S5는 S4의 복잡한 컨볼루션 및 주파수 도메인 접근 방식 대신 효율적이고 널리 구현된 병렬 스캔(parallel scan)을 활용하여 순전히 재귀적으로 시간 도메인에서 작동합니다.
*   **S4와의 수학적 관계 설정:** S5와 S4 간의 수학적 관계를 정립하여 S4의 성공에 핵심적인 HiPPO 초기화(HiPPO initialization) 방식을 S5에 활용할 수 있도록 합니다. 특히, S5는 HiPPO-LegS 행렬의 대각화된 HiPPO-N 근사를 사용합니다.
*   **최첨단 성능 달성:** 여러 장거리 시퀀스 모델링 태스크(Long Range Arena, LRA 벤치마크, Speech Commands 분류 등)에서 S4와 동등하거나 더 나은 최첨단 성능을 달성합니다 (LRA 평균 87.4%, Path-X 태스크 98.5%).
*   **계산 효율성 유지:** S5는 S4와 동일한 계산 복잡도를 유지하며, 시퀀스 길이에 대해 선형적인 복잡도($O(L)$)를 가집니다.
*   **구현 간소화 및 유연성:** S5는 구현이 간단하며, 시간-변화 SSM(time-varying SSM) 및 불규칙하게 샘플링된 관측치(irregularly sampled observations)를 효율적으로 처리할 수 있습니다.

## 📎 Related Works
*   **기존 시퀀스 모델:** RNN(Arjovsky et al., 2016; Erichson et al., 2021), CNN(Bai et al., 2018; Oord et al., 2016), Transformer(Vaswani et al., 2017) 및 이들의 효율적인 변형(Choromanski et al., 2021; Katharopoulos et al., 2020)들이 장거리 시퀀스 모델링에 시도되었습니다.
*   **S4 레이어:** Gu et al. (2021a)이 제안한 S4는 HiPPO 프레임워크(Gu et al., 2020a)와 심층 학습을 결합하여 장거리 시퀀스 모델링에서 최첨단 성능을 달성했습니다.
*   **HiPPO 프레임워크:** Gu et al. (2020a)은 온라인 함수 근사를 위한 HiPPO(High-Order Polynomial Projection Operator) 프레임워크를 제안했으며, S4는 이를 통해 특별히 구성된 상태 행렬로 초기화됩니다.
*   **대각 상태 공간 모델:** DSS(Gupta et al., 2022) 및 S4D(Gu et al., 2022) 레이어는 HiPPO 행렬의 대각 근사(diagonal approximation)를 사용하여 SISO 설정에서 강력한 성능을 보여주었습니다.
*   **병렬 선형 RNN:** Martin & Cundy (2018)는 병렬 스캔을 사용하여 비선형 RNN을 선형 RNN 스택으로 근사하는 연구를 수행했으며, QRNN(Bradbury et al., 2017) 및 SRU(Lei et al., 2018)와 같은 효율적인 RNN이 이 범주에 속합니다.
*   **동시 개발 방법:** Liquid-S4(Hasani et al., 2023) 및 Mega-chunk(Ma et al., 2023)와 같은 방법들이 S5와 비슷한 시기에 개발되었습니다.

## 🛠️ Methodology
S5 레이어는 S4 레이어의 핵심 구성 요소를 단순화하고 재구성합니다.

1.  **S5 구조: SISO에서 MIMO로:**
    *   S4는 $H$개의 독립적인 $N$차원 상태를 가진 SISO SSM 뱅크를 사용하지만, S5는 잠재 상태 크기 $P$와 입력/출력 차원 $H$를 가진 단일 MIMO SSM을 사용합니다.
    *   S5는 선형 변환 후 비선형 활성화 함수를 적용하여 레이어 출력을 생성하며, S4와 달리 추가적인 위치별 선형 혼합 레이어(position-wise linear mixing layer)가 필요하지 않습니다.
    *   S5의 잠재 상태 크기 $P$는 S4의 유효 잠재 상태 크기 $HN$보다 훨씬 작을 수 있습니다.

2.  **S5 매개변수화: 대각화된 동역학:**
    *   효율적인 병렬 스캔을 위해 상태 행렬 $A$가 대각 행렬이 되도록 시스템을 대각화합니다. 연속 시간 상태 행렬을 $A = V\Lambda V^{-1}$로 표현하고, 이를 재매개변수화된 시스템 $\frac{d\tilde{x}(t)}{dt} = \Lambda \tilde{x}(t) + \tilde{B}u(t)$, $y(t) = \tilde{C}\tilde{x}(t) + Du(t)$로 변환합니다. 여기서 $\tilde{x}(t) = V^{-1}x(t)$, $\tilde{B} = V^{-1}B$, $\tilde{C} = CV$입니다.
    *   이 대각화된 시스템은 ZOH(Zero-Order Hold) 방법을 사용하여 이산화됩니다: $\bar{\Lambda} = e^{\Lambda \Delta}$, $\bar{B} = \Lambda^{-1}(\bar{\Lambda}-I)\tilde{B}$.
    *   학습 가능한 매개변수로는 $\tilde{B} \in \mathbb{C}^{P \times H}$, $\tilde{C} \in \mathbb{C}^{H \times P}$, $\text{diag}(D) \in \mathbb{R}^{H}$, $\text{diag}(\Lambda) \in \mathbb{C}^{P}$, 그리고 각 상태별 시간 척도 매개변수 $\Delta \in \mathbb{R}^{P}$가 있습니다.
    *   **초기화:** S4의 HiPPO-LegS 매트릭스는 안정적으로 대각화될 수 없으므로, S5는 HiPPO-N 매트릭스의 대각화를 사용하여 초기화합니다. HiPPO-N은 SISO 시스템에서 HiPPO-LegS와 유사한 동역학을 생성하며, 이 속성은 MIMO 시스템으로 확장됩니다. 켤레 대칭(conjugate symmetry)을 강제하여 실수 출력을 보장하고 런타임을 줄입니다.
    *   HiPPO-N 블록 대각 초기화(block-diagonal initialization)를 통해 S4의 A와 $\Delta$를 개별적으로 학습하는 유연성을 확보할 수 있습니다.

3.  **S5 계산: 완전 재귀적:**
    *   S5는 전체 시퀀스를 사용할 수 있을 때 효율적인 병렬 스캔을 활용하여 재귀적으로 작동합니다. 이는 온라인 생성과 오프라인 처리 모두에 적용될 수 있습니다.
    *   병렬 스캔은 시간-변화 SSM 및 불규칙하게 샘플링된 시계열 데이터 처리에도 효율적입니다. S4의 컨볼루션 방식은 시간 불변 시스템과 규칙적인 간격의 관측치를 요구합니다.
    *   S5는 S4와 동일한 계산 복잡도(런타임 및 메모리)를 가집니다. $P=O(H)$일 때, 병렬 오프라인 처리에서 $O(H^2L + HL)$ 연산, 온라인 생성에서 스텝당 $O(H^2)$ 연산이 필요합니다.

## 📊 Results
S5 레이어는 S4 레이어 및 다른 기준 방법과 비교하여 여러 벤치마크에서 최첨단 성능을 달성합니다.

*   **Long Range Arena (LRA) 벤치마크:**
    *   S5는 선형 복잡도 모델 중 가장 높은 평균 점수인 **87.4%**를 달성했습니다.
    *   특히, 가장 긴 시퀀스 길이를 가진 Path-X 태스크에서 모든 모델 중 가장 높은 **98.5%** 정확도를 기록했습니다.

*   **Raw Speech Classification (Speech Commands):**
    *   S5는 기존 S4 방법들을 능가하고 동시 개발된 Liquid-S4와 유사한 성능을 보였습니다 (**96.52%**).
    *   16kHz로 학습된 모델을 8kHz 데이터에 추가 미세 조정 없이 적용하는 제로샷(zero-shot) 테스트에서 S5는 **94.53%**의 정확도를 달성하여 기준 모델보다 우수했습니다.

*   **Variable Observation Interval (Pendulum Regression):**
    *   불규칙하게 샘플링된 데이터 처리 능력: S5는 회귀 태스크에서 CRU보다 낮은 평균 오차(**3.41e-3**)를 달성하며 우수한 성능을 보였습니다.
    *   S5는 CRU보다 훨씬 빠른 속도(약 86배)를 보여주었습니다.

*   **Pixel-level 1-D Image Classification (sMNIST, psMNIST, sCIFAR):**
    *   S5는 sMNIST (**99.65%**), psMNIST (**98.67%**), sCIFAR (**90.10%**)에서 S4와 거의 동일한 성능을 보여주며, 여러 RNN 기반 최첨단 방법들을 능가했습니다.

## 🧠 Insights & Discussion
*   **효율성 및 성능 균형:** S5는 S4의 계산 효율성을 유지하면서 최첨단 성능을 달성하는 단순하고 일반적인 상태 공간 레이어임을 입증했습니다.
*   **유연성 증가:** S5의 병렬 스캔 방식은 컨볼루션 기반 S4와 달리, 시간-변화 SSM(예: 입력에 따라 매개변수가 변하는 시스템)과 불규칙하게 샘플링된 시계열 데이터를 효율적으로 처리할 수 있는 새로운 가능성을 열었습니다. 이는 펜듈럼 회귀 태스크에서 입증되었습니다.
*   **HiPPO 초기화의 중요성:** 어블레이션 연구는 S5의 강력한 성능이 연속 시간 매개변수화(continuous-time parameterization)와 HiPPO-N 초기화에 크게 기인한다는 것을 보여줍니다. 특히 HiPPO 초기화가 없는 경우 성능이 크게 저하되었습니다.
*   **단순성과 확장성:** S5 레이어의 단순하고 일반적인 MIMO SSM 설계는 고전적인 확률론적 상태 공간 모델링과의 연결뿐만 아니라, 필터링 및 스무딩 연산의 병렬화와 같은 더 광범위한 응용 분야로의 확장을 가능하게 합니다.
*   **잠재적 개선 사항:** 시간 척도 매개변수의 개별 학습(각 상태별 $P$개의 $\Delta$)과 블록 대각 초기화가 성능 향상에 기여했음을 확인했습니다. 이는 모델이 데이터의 다양한 시간 척도를 포착하도록 돕는 것으로 해석됩니다.

## 📌 TL;DR
S5는 S4의 단일 입력, 단일 출력(SISO) SSM 뱅크를 단일 다중 입력, 다중 출력(MIMO) SSM으로 대체하여 장거리 시퀀스 모델링을 단순화합니다. S5는 효율적인 병렬 스캔을 활용하여 S4의 복잡한 주파수 도메인 컨볼루션 방식을 대체하고, HiPPO-N 행렬로 초기화하여 강력한 성능을 유지합니다. 결과적으로 S5는 S4와 동등한 계산 효율성을 가지면서 LRA 벤치마크 및 음성 분류와 같은 여러 장거리 시퀀스 태스크에서 최첨단 성능을 달성하며, 불규칙하게 샘플링된 데이터 처리와 같은 새로운 활용 가능성을 제시합니다.