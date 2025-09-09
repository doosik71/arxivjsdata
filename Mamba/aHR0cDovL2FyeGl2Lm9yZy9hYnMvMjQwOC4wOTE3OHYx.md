# MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Model
Changcheng Xiao, Qiong Cao, Zhigang Luo, Long Lan

## 🧩 Problem to Solve
다중 객체 추적(MOT) 분야에서 지배적인 패러다임인 '추적-by-검출(tracking-by-detection)' 방법은 일반적으로 칼만 필터(Kalman Filter)를 사용하여 선형 객체 모션을 가정하고 미래 위치를 예측합니다. 그러나 댄스나 스포츠와 같이 비선형적이고 다양한 움직임을 보이는 시나리오에서는 이러한 방법의 성능이 저하됩니다. 또한, MOT에서 학습 기반 모션 예측기를 활용하는 데 대한 연구가 제한적이었습니다. 더욱이, 객체는 가려짐(occlusion)이나 모션 블러로 인해 감지되지 않아 궤적이 조기에 종료되는 문제가 발생합니다.

## ✨ Key Contributions
*   복잡한 시나리오에서 다양한 모션 패턴을 모델링하도록 설계된 데이터 기반 모션 예측기, MTP(Mamba moTion Predictor)를 제안합니다.
*   손실된 트랙렛을 재확립하기 위해 MTP를 자기회귀 방식으로 사용하는 트랙렛 패칭 모듈(Tracklet Patching Module, TPM)을 제안합니다.
*   설계된 MTP와 TPM을 갖춘 온라인 트래커인 MambaTrack은 복잡한 댄스 및 스포츠 시나리오에서 어려운 데이터 연관(data association) 문제를 효과적으로 처리합니다.
*   모션 기반 온라인 트래커로서 MambaTrack은 DanceTrack [42] 및 SportsMOT [9] 벤치마크에서 최첨단(state-of-the-art) 성능을 달성합니다.

## 📎 Related Works
*   **추적-by-검출(Tracking-by-detection)**: SORT [3], ByteTrack [54], FairMOT [55]와 같은 방법은 일반적으로 검출기와 재식별 기술의 발전에 의존하며, 구별 가능한 외모와 규칙적인 모션 패턴을 가진 시나리오에 효과적입니다.
*   **모션 모델(Motion Models)**:
    *   **필터 기반**: 칼만 필터(Kalman Filter, KF) [22]를 사용하는 SORT [3]와 같은 방법은 규칙적인 모션에 적합하지만 복잡한 모션 시나리오에서는 어려움을 겪습니다. OC_SORT [6]는 KF의 한계를 개선하여 비선형 모션 및 가려짐을 처리합니다.
    *   **학습 기반**: Tracktor [2]는 변위 예측을 위한 회귀 브랜치를, CenterTrack [57]은 중심 오프셋 예측을, ArTIST [40]는 객체 상호작용을 확률 분포로 모델링합니다. 이들은 계산 비용이 높거나 복잡한 훈련 절차를 요구합니다.
*   **상태 공간 모델(State Space Models, SSMs)**: HiPPO [15], S4 [16], S5 [41]와 같은 초기 연구부터 Mamba [14]에 이르기까지 순차 데이터 모델링에서 효율성과 성능을 개선해왔습니다. Mamba는 선택적 메커니즘을 통합하여 긴 시퀀스 모델링에서 선형 시간에 가까운 복잡도를 제공합니다.
*   **Mamba의 MOT 적용**: Huang et al. [19]는 Mamba 블록을 사용하여 객체 모션 패턴을 모델링했습니다. 본 연구는 Bi-Mamba 인코딩 레이어와 트랙렛 패칭 모듈을 통해 차별화됩니다.

## 🛠️ Methodology
MambaTrack은 추적-by-검출 패러다임을 따르며, YOLOX [12]와 같은 기성 검출기를 사용하여 바운딩 박스를 얻습니다. 이 방법은 트랙렛을 $T_{\text{active}}$ (활성)와 $T_{\text{lost}}$ (손실)로 분리하여 처리합니다.

### 1. Mamba Motion Predictor (MTP)
MTP는 객체의 모션 패턴을 모델링하고 다음 움직임을 예측하기 위해 Mamba 아키텍처를 활용합니다.
*   **시간 토큰화 계층 (Temporal Tokenization Layer)**:
    *   트랙렛의 과거 궤적 $O_{\text{in}} = [o_{t-q}, o_{t-q+1}, \dots, o_{t-1}] \in \mathbb{R}^{q \times 4}$를 입력으로 사용합니다. 여기서 $o = [\delta c_x, \delta c_y, \delta w, \delta h]$는 바운딩 박스 중심, 너비, 높이의 정규화된 변화량을 나타냅니다.
    *   단일 선형 계층을 통해 입력 토큰 시퀀스 $X = \text{Embedding}(O_{\text{in}})$를 얻습니다. 여기서 $X \in \mathbb{R}^{q \times d_m}$입니다.
*   **Bi-Mamba 인코딩 계층 (Bi-Mamba Encoding Layer)**:
    *   $L$개의 Bi-Mamba 블록으로 구성되며, 객체 궤적 정보를 완전히 활용하고 Mamba의 단방향 한계를 극복하기 위해 순방향 및 역방향 Mamba 모듈을 포함합니다.
    *   $l$-번째 Bi-Mamba 블록의 추론 과정은 다음과 같습니다:
        $$
        \hat{X}_{\text{forward}} = \text{Mamba}(X_{l-1}) \\
        \hat{X}_{\text{backward}} = \text{Mamba}_{\text{backward}}(X_{l-1}) \\
        \hat{Y} = \hat{X}_{\text{forward}} + \hat{X}_{\text{backward}} \\
        X_l = \hat{Y} + \text{LN}(\text{MLP}(\hat{Y}))
        $$
*   **예측 헤드 및 훈련 (Prediction Head and Training)**:
    *   Bi-Mamba 인코딩 계층의 출력을 평균 풀링 계층으로 집계한 후, 두 개의 완전 연결 계층으로 구성된 예측 헤드가 프레임 간 바운딩 박스 오프셋 $\hat{O}$를 예측합니다.
    *   훈련에는 Smooth L1 손실이 사용됩니다: $L(\hat{O}, O^*) = \frac{1}{4} \sum \text{smooth}_{L_1}(\hat{\delta}_i - \delta_i)$, 여기서 $i \in \{c_x, c_y, w, h\}$입니다.

### 2. 트랙렛 패칭 모듈 (Tracklet Patching Module, TPM)
*   가려짐이나 검출기 실패로 인해 누락된 관측 지점을 보상하고 궤적의 일관성을 높이는 것을 목표로 합니다.
*   손실된 트랙렛이 업데이트를 받지 못할 경우, MTP를 자기회귀 방식으로 사용하여 이전 프레임의 예측 바운딩 박스를 실제 관측으로 간주하고, 이를 기반으로 현재 프레임의 위치를 예측합니다.
*   지속적으로 매칭에 실패하면, MTP를 사용하여 과거 궤적 시퀀스와 예측된 바운딩 박스를 활용하여 미래 바운딩 박스 $\hat{p}_{t+1}^i = \text{MTP}(T_{\text{past}}, \hat{p}_t^i)$를 프레임별로 예측합니다.

### 3. 추론 과정 (Inference)
1.  **첫 번째 매칭**: 활성 트랙 $T_{\text{active}}$의 예측 바운딩 박스 $\hat{B}_t$와 현재 프레임의 검출 $B_t$를 IoU 유사성을 기반으로 헝가리안 알고리즘 [24]으로 매칭합니다.
2.  **두 번째 매칭 (손실된 트랙렛 재확립)**: 남아있는 검출 $B_u$를 손실된 트랙렛 $T_{\text{lost}}$ 및 매칭되지 않은 트랙렛 $T_u$의 예측 바운딩 박스 $\hat{P}$와 매칭하여 손실된 트랙렛을 재확립합니다.
3.  **트랙렛 업데이트**: 매칭된 트랙렛은 새 관측으로 업데이트되고, 매칭되지 않은 손실된 트랙렛은 마지막 예측 바운딩 박스로 업데이트됩니다.
4.  **새로운 트랙렛 초기화**: 신뢰도 점수가 $t_{\text{thresh}}=0.6$보다 높은 매칭되지 않은 검출은 새로운 트랙렛으로 초기화됩니다.
5.  **트랙렛 종료**: $t_{\text{terminate}}=30$ 프레임 동안 업데이트를 받지 못한 손실된 트랙은 종료됩니다.

## 📊 Results
본 연구는 DanceTrack [42] 및 SportsMOT [9] 데이터셋에서 MambaTrack의 성능을 평가했습니다.

*   **DanceTrack 벤치마크**:
    *   MambaTrack은 핵심 지표인 HOTA에서 56.8%를 달성하여 최첨단 방법인 OC_SORT를 2.2%p 앞섰습니다.
    *   IDF1 점수 또한 57.8로 최고치를 기록하며, 궤적 일관성 측면에서 2위보다 3.2%p 우수했습니다.
*   **SportsMOT 벤치마크**:
    *   MambaTrack은 모든 지표에서 모션 정보에만 의존하는 다른 추적 알고리즘을 능가했습니다.
    *   ByteTrack 대비 HOTA에서 거의 10%p, IDF1에서 3%p, AssA에서 9.1%p 높은 성능을 보였습니다.
    *   향상된 칼만 필터 기반 접근 방식인 OC-SORT보다도 우수한 성능을 입증했습니다.
*   **Ablation Study**:
    *   **MTP의 효과**: 제안된 MTP는 베이스라인(Kalman Filter) 대비 HOTA에서 9%p, IDF1에서 3.6%p, AssA에서 7.8%p의 상당한 개선을 가져왔습니다. 이는 MTP가 객체의 비선형 모션을 효율적으로 모델링함을 보여줍니다.
    *   **TPM의 효과**: TPM 모듈 도입 시 IDF1 1.6%p, AssA 0.7%p 추가 개선을 통해 궤적 일관성이 향상되었습니다.
    *   **다른 모션 모델과의 비교**: MTP는 Kalman Filter, LSTM, Transformer와 같은 다른 학습 기반 모션 예측기보다 모든 지표에서 최적의 결과를 달성했습니다.
    *   **Bi-Mamba 인코딩 계층 설계**: Vanilla Mamba 대비 Bi-Mamba가 HOTA 2.5%p, AssA 3.3%p, IDF1 2.3%p 더 높은 성능을 보여, Bi-Mamba의 효과를 검증했습니다.
    *   **Bi-Mamba 블록 수**: $L=3$일 때 MTP는 HOTA, IDF1, AssA, MOTA에서 최적의 성능을 보였습니다.
*   **추론 시간 분석**: DanceTrack 검증 세트에서 단일 프레임 처리 시간은 67ms (17 FPS)였으며, 트래킹 구성 요소는 전체 추론 시간의 19% (11.37ms)만을 차지하여 주요 계산 부담이 아님을 보여주었습니다.

## 🧠 Insights & Discussion
MambaTrack은 Mamba 기반의 상태 공간 모델을 다중 객체 추적에 성공적으로 통합하여, 특히 복잡하고 비선형적인 움직임과 외모의 유사성으로 인해 기존 칼만 필터 기반 방법론이 어려움을 겪는 시나리오(댄스, 스포츠)에서 강력한 대안을 제시합니다. MTP는 시공간적 역학을 효과적으로 학습하여 정확한 다음 위치를 예측하며, TPM은 MTP의 자기회귀적 활용을 통해 단기적인 가려짐이나 검출 실패로 인한 궤적 손실을 보상하여 궤적의 일관성을 크게 향상시킵니다. 이 연구는 데이터 기반 모션 모델의 잠재력과 SSM의 효능을 입증하며, 간단하고 직관적인 접근 방식임에도 불구하고 최첨단 성능을 달성하여 향후 모션 기반 추적 알고리즘 개발을 위한 강력한 기준선을 제공합니다. 주요 병목은 트래킹 구성 요소가 아닌 검출 구성 요소에 있음을 확인하여, 트래킹 효율성은 높다는 점을 시사합니다.

## 📌 TL;DR
기존 다중 객체 추적(MOT) 방법의 한계(비선형 모션, 가려짐)를 극복하기 위해, 본 논문은 Mamba 기반의 데이터 구동 모션 예측기(MTP)를 제안합니다. MTP는 객체의 복잡한 모션 패턴을 학습하고 다음 위치를 예측하며, 트랙렛 패칭 모듈(TPM)은 MTP를 자기회귀 방식으로 활용하여 누락된 관측치를 보상하고 궤적 일관성을 높입니다. MambaTrack이라는 이름의 이 추적기는 DanceTrack 및 SportsMOT 벤치마크에서 HOTA 및 IDF1과 같은 핵심 지표에서 최첨단 성능을 달성하며, 기존 칼만 필터 및 다른 학습 기반 모션 예측기보다 뛰어난 정확도와 궤적 일관성을 입증합니다.