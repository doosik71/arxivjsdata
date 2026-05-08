# A2VIS: Amodal-Aware Approach to Video Instance Segmentation

Minh Tran, Thang Pham, Winston Bounsavy, Tri Nguyen, Ngan Le (2025)

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 및 다중 객체 추적(Multiple Object Tracking, MOT) 작업에서 발생하는 **폐색(Occlusion)** 문제를 해결하고자 한다. 비디오 내에서 객체가 다른 객체에 의해 부분적으로 또는 완전히 가려지는 폐색 현상은 객체의 가시적 영역(visible part)만을 처리하는 기존 방법론들에서 심각한 성능 저하를 야기한다. 특히, 긴 시퀀스에서 객체가 사라졌다가 다시 나타날 때, 기존 방식들은 객체의 동일성을 유지하지 못하고 **ID 스위칭(Identity Switch)**이 빈번하게 발생하는 한계가 있다.

따라서 본 연구의 목표는 객체의 가려진 부분까지 포함하여 전체 형태를 추론하는 **Amodal Representation**을 VIS 프레임워크에 통합함으로써, 폐색 상황에서도 안정적으로 객체를 식별하고 추적할 수 있는 A2VIS(Amodal-Aware Video Instance Segmentation) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 지각 능력과 유사하게, 가려진 부분까지 포함한 전체 형태를 인식하는 Amodal Segmentation을 통해 시공간적 차원에서 객체 정보의 안정적인 흐름을 확보하는 것이다. 가시적 영역(visible segmentation)은 폐색 시 급격하게 변화하지만, Amodal 영역은 상대적으로 변화가 적어 시간축을 따라 더 높은 일관성을 유지할 수 있다는 직관에 기반한다.

주요 기여 사항은 다음과 같다.

1. **A2VIS 프레임워크 제안**: 검출, 분할, 추적 과정 전체에 Amodal 특성을 통합하고, 비디오 전체의 가시적/Amodal 특성을 모두 캡처하는 **Global Instance Prototypes**를 도입하여 폐색 상황에서의 강건한 객체 업데이트 및 연관성을 구현하였다.
2. **SAMH(Spatiotemporal-prior Amodal Mask Head) 개발**: 단기적 정보(인접 프레임의 가시적 영역)와 장기적 정보(비디오 전체의 Amodal 영역)를 모두 활용하는 시공간 사전 지식 기반의 마스크 헤드를 설계하였다.
3. **성능 우위 입증**: 다양한 벤치마크 실험을 통해 기존 SOTA VIS 및 MOT 방법론 대비 객체의 전체 형태를 더 잘 이해하고 추적함을 정량적, 정성적으로 증명하였다.

## 📎 Related Works

### 1. Amodal Segmentation

기존의 Amodal Segmentation 연구는 주로 이미지 기반으로 진행되었으며, 가려진 영역을 예측하는 마스크 헤드나 확산 모델(Diffusion Models)을 이용한 인페인팅(Inpainting) 방식이 사용되었다. 비디오 기반 Amodal 연구(예: SaVos)는 LSTM이나 뷰 퓨전(View Fusion)을 통해 시간적 일관성을 확보하려 했으나, 대부분 다단계(multi-stage) 프레임워크로 구성되어 있었다. 반면, A2VIS는 검출, 추적, 가시적/Amodal 분할을 동시에 수행하는 **End-to-End** 프레임워크라는 점에서 차별된다.

### 2. Video Instance Segmentation (VIS)

초기 VIS는 프레임별 예측 후 후처리로 연관성을 찾는 방식이었으나, 최근에는 VITA나 GenVIS와 같이 인스턴스 프로토타입(Instance Prototype)을 사용하는 방식으로 발전하였다. 하지만 이러한 최신 기법들조차 기본적으로 가시적 요소(visible elements) 처리에 의존하며, 폐색 시 객체의 전체적인 이해가 부족하여 ID 스위칭 문제에서 자유롭지 못하다. A2VIS는 Global-Local 프로토타입 전략과 SAMH 모듈을 통해 이 문제를 해결한다.

### 3. Multi-Object Tracking (MOT)

MOT는 주로 Bounding Box 기반의 추적 방식을 사용하는데, 이는 객체들이 겹칠 때 모호성이 증가하는 문제가 있다. A2VIS는 Bounding Box 대신 Amodal Segmentation을 사용함으로써 인스턴스 간의 구분을 더 명확히 하고, 가려진 상황에서도 전체 인스턴스를 인식하여 일관된 추적을 가능하게 한다.

## 🛠️ Methodology

### 전체 시스템 구조

A2VIS의 파이프라인은 입력 비디오를 여러 클립($V_k$)으로 나누어 처리하며, 전체 과정은 다음과 같이 구성된다:

1. **Instance Prototype Modelling**: VITA 구조를 채택하여 클립 단위의 인스턴스 프로토타입 $p^k$와 프레임 특징 $F^k$를 생성한다.
2. **Instance Prototype Update**: 클립 단위 프로토타입 $p^k$를 사용하여 비디오 전체를 대표하는 **Global Instance Prototypes** $p^G$를 업데이트한다. 이를 통해 새로운 객체를 추가하고 기존 객체와의 연관성을 유지한다.
3. **Visible Mask Head**: $p^G$와 $F^k$를 이용하여 가시적 분할 결과 $M^k$를 생성한다.
4. **SAMH (Spatiotemporal-prior Amodal Mask Head)**: 가시적 분할 $M^k$와 $p^G$, $F^k$를 입력으로 받아 Amodal 분할 $A^k$를 예측하고 $p^G$를 다시 업데이트한다.
5. **Classification Head**: 최종적으로 $p^G$를 통해 각 인스턴스의 카테고리를 예측한다.

### SAMH의 상세 동작 및 주요 방정식

SAMH는 객체의 가려진 부분을 예측하기 위해 두 가지 시공간 사전 정보(Spatiotemporal Prior)를 활용한다.

- **단기 정보 (Short-range)**: 인접 프레임에서 나타나는 가시적 영역 정보 ($\text{VSPM, Visible Spatiotemporal-prior Mask}$).
- **장기 정보 (Long-range)**: 이전 클립들로부터 누적된 Amodal 분할 정보 ($\text{ASPM, Amodal Spatiotemporal-prior Mask}$).

이 두 정보를 통합한 시공간 사전 마스크 $T^k$를 생성하며, 이를 Masked Attention 메커니즘에 적용하여 $p^G$를 디코딩한다. 구체적인 수식은 다음과 같다.

$$ p^G_l = \text{softmax}(T^k + QK^\top)V + p^G_{l-1} $$
여기서 $Q, K, V$는 각각 다음과 같이 정의된다:
$$ Q = p^G_l \cdot W^Q, \quad K = O^k \cdot W^K, \quad V = O^k \cdot W^V $$
($O^k$는 Amodal Feature Extraction $\Omega$를 통해 추출된 amodal attention feature이다.)

### 인스턴스 프로토타입 업데이트

Global Prototype $p^G$는 Cross-Attention 메커니즘을 통해 클립 단위의 $p^k$로부터 정보를 업데이트한다.
$$ Z = (W^{Q'} p^G)^\top \cdot K'(p^k) $$
$$ p^G = p^G + Z W^{V'} p^k $$

### 손실 함수 (Loss Function)

모델은 Hungarian matching 알고리즘을 사용하여 예측값과 Ground Truth(GT) 간의 최적 할당 $\hat{\sigma}$를 찾는다. 최적화 대상이 되는 비용 함수는 다음과 같다.
$$ \hat{\sigma} = \arg \min_{\sigma \in S^N} \sum_{i=1}^N \left[ -\log \hat{c}_{\sigma(i)}(c^{gt}_i) + \mathbb{1}_{c^{gt}_i \neq \emptyset} (L_v + L_a) \right] $$
여기서 $L_v$는 가시적 마스크 손실, $L_a$는 Amodal 마스크 손실이며, 둘 다 Binary Cross Entropy(BCE)를 사용한다. 최종 손실 $L_{final}$은 할당된 $\hat{\sigma}$를 바탕으로 분류 손실과 두 마스크 손실의 합으로 계산된다.

## 📊 Results

### 실험 설정

- **데이터셋**: FISHBOWL (시뮬레이션 수족관), SAIL-VOS (GTA-V 기반 합성 데이터)를 사용하였다.
- **측정 지표**:
  - Segmentation Tracking: $\text{AP}, \text{AR}$
  - Bounding Box Tracking: $\text{HOTA}, \text{IDF1}, \text{IDs}$ (ID Switch)
- **Backbone**: ResNet-50 및 Swin-L을 사용하였다.

### 주요 결과

1. **SOTA VIS 방법론 비교**: A2VIS는 모든 백본과 데이터셋에서 기존 SOTA 모델인 GenVIS를 상회하는 성능을 보였다. 특히 **IDF1**과 **IDs** 지표에서 큰 향상을 보였는데, 이는 Amodal 인식 능력이 추적의 일관성을 획기적으로 높였음을 의미한다.
2. **Amodal VIS 베이스라인 비교**: 가시적 감독을 Amodal 감독으로 대체한 기존 모델들(VITA-Amodal, GenVIS-Amodal)보다 월등한 성능을 기록하였다. 이는 단순한 감독 신호의 변경보다 SAMH와 같은 구조적 접근이 Amodal 예측에 필수적임을 시사한다.
3. **MOT 방법론 비교**: Bounding Box 기반의 TrackFormer, MOTRv2보다 우수한 성능을 보였으며, 이는 Amodal Segmentation이 겹침으로 인한 모호성을 제거하는 데 효과적임을 보여준다.
4. **Ablation Study**:
    - **VSPM & ASPM**: 두 사전 정보가 모두 있을 때 최고의 성능을 냈으며, 이는 단/장기 정보의 상호보완적 역할이 중요함을 입증한다.
    - **클립 길이 ($N_c$)**: $N_c=3$일 때 가장 높은 성능을 보였다. $N_c=1$일 경우 시공간 사전 정보를 활용할 수 없어 성능이 크게 하락하였다.
    - **폐색 수준별 분석**: 폐색률이 50% 이상인 가혹한 환경에서도 GenVIS-Amodal 대비 높은 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

A2VIS의 가장 큰 강점은 **Amodal Awareness를 통한 추적 안정성 확보**이다. 기존 VIS 모델들이 폐색 시 객체를 잃어버리고 다시 나타났을 때 새로운 ID를 부여하는 것과 달리, A2VIS는 가려진 영역의 형태를 예측함으로써 객체의 궤적을 연속적으로 인식한다. 이는 단순한 마스크 정밀도 향상을 넘어, 비디오 이해의 차원을 가시적 영역에서 객체 전체의 존재론적 영역으로 확장한 것으로 해석할 수 있다.

### 한계 및 미해결 과제

1. **형태 변화의 취약성**: 본 모델은 인접 프레임의 가시적 힌트를 통해 가려진 부분을 복원하므로, 객체 자체가 급격하게 형태를 바꾸는 경우(Intrinsic shape change) 예측 정확도가 떨어질 수 있다.
2. **프레임 외 폐색**: 현재 모델은 프레임 내의 폐색(In-frame occlusion)만을 다룬다. 객체가 프레임 밖으로 완전히 나갔다가 다시 들어오는 경우에 대한 Ground Truth가 부족하여 이를 명시적으로 처리하지 못하며, Amodal 마스크를 프레임 크기 내로 제한하고 있다.
3. **실제 데이터셋 부족**: 합성 데이터셋(FISHBOWL, SAILVOS) 위주로 검증되었으며, 실제 환경(Real-world)의 Amodal VIS 데이터셋이 부족하여 제로샷(Zero-shot) 평가에 의존해야 했다.

## 📌 TL;DR

A2VIS는 비디오 인스턴스 분할에서 고질적인 문제인 **폐색으로 인한 ID 스위칭**을 해결하기 위해, 객체의 가려진 부분까지 예측하는 **Amodal Representation**을 도입한 프레임워크이다. 특히 단/장기 시공간 정보를 통합하는 **SAMH 모듈**과 비디오 전체를 아우르는 **Global Instance Prototypes**를 통해 폐색 상황에서도 객체의 동일성을 유지하며 정밀한 분할 및 추적을 수행한다. 이 연구는 향후 자율주행이나 보안 관제와 같이 객체 겹침이 빈번한 실제 환경의 비디오 분석 연구에 중요한 기반이 될 가능성이 높다.
