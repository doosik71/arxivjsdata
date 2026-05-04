# Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement

Beomyoung Kim, YoonJoon Yoo, Chaeeun Rhee, Junmo Kim (Year not explicitly mentioned in the provided text)

## 🧩 Problem to Solve

본 논문은 이미지 레벨 레이블(image-level labels)만을 사용하는 약지도 인스턴스 분할(Weakly-Supervised Instance Segmentation, WSIS)의 한계를 해결하고자 한다. WSIS는 약지도 시맨틱 분할(WSSS)보다 훨씬 더 어려운 과제인데, 그 이유는 이미지 레벨 레이블에서 각 인스턴스의 개별적인 위치 정보(instance-wise localization)를 추출하기 어렵기 때문이다.

기존의 WSIS 접근 방식들은 주로 MCG(Multiscale Combinatorial Grouping)나 Salient Instance Segmentor와 같은 기성 제안 기술(off-the-shelf proposal techniques)에 의존하여 인스턴스 마스크 후보를 생성한다. 그러나 이러한 방식은 다음과 같은 두 가지 심각한 문제를 야기한다. 첫째, 이러한 제안 기술들은 객체 경계나 클래스 불가지론적(class-agnostic) 인스턴스 마스크와 같은 고수준 레이블로 사전 학습되어야 하므로, 엄격한 의미의 '이미지 레벨 지도 학습' 설정에서 벗어나 있으며, 일반적인 객체 위주로 학습되어 의료 영상과 같은 특수 도메인에 적용하기 어렵다. 둘째, 가짜 레이블(pseudo-labels)에서 인스턴스가 누락될 경우, 해당 영역이 배경(background) 클래스로 학습되어 인스턴스와 배경 사이의 혼동이 발생하는 '시맨틱 드리프트(semantic drift)' 문제가 발생하여 성능이 저하된다.

결과적으로 본 논문의 목표는 기성 제안 기술 없이 이미지 레벨 레이블만을 사용하여 엄격한 WSIS 설정을 따르면서, 시맨틱 드리프트 문제를 해결하여 인스턴스 분할 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문은 **BESTIE**라는 새로운 WSIS 프레임워크를 제안하며, 핵심 아이디어는 다음과 같다.

1. **시맨틱 지식 전이(Semantic Knowledge Transfer):** 사전 학습된 제안 기술 없이, 상대적으로 연구가 많이 진행된 WSSS의 지식을 WSIS로 전이하여 초기 가짜 인스턴스 레이블(pseudo instance labels)을 생성한다.
2. **Peak Attention Module (PAM):** 이미지 레벨 레이블에서 더 정확한 인스턴스 힌트(instance cues)를 추출하기 위해, 활성화 맵(CAM)에서 노이즈를 제거하고 객체의 대표적인 정점 영역(peak region)을 강조하는 PAM을 제안한다.
3. **자기 정제(Self-Refinement):** 생성된 가짜 레이블의 오류(false-negatives)를 자기 지도 학습 방식으로 정제하여 온라인 방식으로 학습에 반영함으로써 레이블의 품질을 점진적으로 향상시킨다.
4. **인스턴스 인식 가이드(Instance-Aware Guidance, IAG):** 시맨틱 드리프트 문제를 해결하기 위해, 가이드 영역을 레이블이 지정된 인스턴스 영역으로만 동적으로 할당하여 배경과 인스턴스 간의 혼동을 방지한다.

## 📎 Related Works

### 1. 약지도 시맨틱 분할 (WSSS)

대부분의 WSSS 연구는 Class Activation Maps (CAMs)를 사용하여 클래스별 영역을 국지화한다. CAM은 주로 판별력이 높은 일부 영역에만 집중하는 경향이 있어, 이를 확장하기 위해 AE-PSL, DRS 등의 방법이 제안되었다. 본 논문은 이러한 WSSS의 성과를 WSIS로 전이하는 전략을 사용한다.

### 2. 인스턴스 분할 (Instance Segmentation)

Mask R-CNN과 같은 박스 기반 2단계 방법이 주류였으나, 최근에는 Panoptic-DeepLab과 같이 센터 맵(center map)과 오프셋 벡터(offset vectors)를 사용하는 박스 없는 1단계 방법들이 제안되었다. 본 논문은 후자의 표현 방식을 채택한다.

### 3. 약지도 인스턴스 분할 (WSIS)

PRM, LIID, Mask R-CNN 기반의 정제 방법 등이 존재한다. 하지만 이들은 MCG나 Salient Instance Segmentor 같은 외부 제안 기술에 의존하며, 이는 본 논문이 지적한 '이미지 레벨 지도 학습의 정의 위배' 및 '도메인 확장성 부족' 문제를 안고 있다. IRN과 같은 제안 기술 없는(proposal-free) 방법이 있으나, 인스턴스 간 관계가 아닌 클래스 간 관계에 기반하여 정확도가 떨어지며 시맨틱 드리프트 문제를 고려하지 않았다.

## 🛠️ Methodology

### 1. 인스턴스 표현 (Instance Representation)

본 논문은 각 인스턴스를 하나의 센터 포인트와 그 포인트로 향하는 2D 오프셋 벡터로 표현한다. 네트워크는 시맨틱 맵, 센터 맵, 오프셋 맵의 세 가지 브랜치를 출력한다. 픽셀 $(i, j)$의 인스턴스 ID $k_{i,j}$는 다음과 같이 결정된다.

$$k_{i,j} = \text{argmin}_k ||(x_k, y_k) - ((i,j) + O(i,j))||$$

여기서 $(x_k, y_k)$는 $k$번째 센터 포인트의 좌표이며, $O(i,j)$는 해당 픽셀의 오프셋 벡터이다.

### 2. 시맨틱 지식 전이 (Semantic Knowledge Transfer)

WSSS의 결과와 PAM을 통해 추출된 인스턴스 힌트를 결합하여 가짜 인스턴스 마스크를 생성한다.

- **절차:** WSSS 출력 결과에 Connected Component Labeling (CCL) 알고리즘을 적용하여 마스크 후보군을 생성한다. 각 후보 마스크 내에 정확히 하나의 인스턴스 힌트(PAM 결과)가 포함된 경우에만 이를 유효한 가짜 인스턴스 마스크로 선택한다.
- **Peak Attention Module (PAM):** CAM의 노이즈를 줄이기 위해 설계되었다.
  - **Selector:** 글로벌 맥스 풀링을 통해 정점 영역의 기준점 $S_p$를 선택한다.
  - **Controller:** 정점 영역을 얼마나 강화할지 결정하는 제어 값 $G_p$ (상수 $\alpha=0.7$)를 설정한다.
  - **Peak Stimulator:** $\tau_p = S_p \cdot G_p$를 임계값으로 설정하여, 이보다 낮은 값의 노이즈 영역을 0으로 만들어 제거한다.

### 3. 인스턴스 인식 가이드 (Instance-Aware Guidance, IAG)

시맨틱 드리프트 문제를 해결하기 위해, 오프셋 맵과 센터 맵의 손실 함수를 계산할 때 레이블이 지정된 인스턴스 영역에 대해서만 가이드를 제공한다. 즉, 가짜 레이블에서 누락된 인스턴스 영역은 손실 함수 계산에서 제외하여, 네트워크가 배경을 인스턴스로 오인하여 학습하는 것을 방지한다.

### 4. 자기 지도 가짜 레이블 정제 (Self-Refinement)

초기 가짜 레이블의 낮은 정밀도(VOC 2012 기준 true-positives 약 30%)를 개선하기 위해 온라인 정제 과정을 거친다.

- **절차:** IAG를 통해 안정적으로 학습된 네트워크의 출력물을 사용하여 인스턴스 그룹화를 수행하고, 이를 통해 더 정밀한 정제된 레이블(refined label)을 생성하여 다시 학습에 반영한다.
- **Center Clustering:** 오프셋 맵의 벡터 크기가 0에 가까운 지점을 찾아 센터 포인트를 보완하는 알고리즘을 적용한다.

### 5. 손실 함수 (Loss Functions)

최종 손실 함수 $L$은 다음과 같이 세 가지 손실의 가중합으로 정의된다.

$$L = \lambda_{\text{center}} L_{\text{center}} + \lambda_{\text{offset}} L_{\text{offset}} + \lambda_{\text{sem}} L_{\text{sem}}$$

- **센터 맵 손실 ($L_{\text{center}}$):** 가중치 마스크 $W(i,j)$를 적용한 MSE 손실이다. $W(i,j)$는 정제된 레이블의 신뢰도 점수로 작동한다.
$$L_{\text{center}} = \frac{1}{|P_{\text{pseudo}}|} \sum_{(i,j) \in P_{\text{pseudo}}} (C(i,j) - \hat{C}(i,j))^2 + \frac{1}{|P_{\text{refined}}|} \sum_{(i,j) \in P_{\text{refined}}} W(i,j) \cdot (C(i,j) - \bar{C}(i,j))^2$$
- **오프셋 맵 손실 ($L_{\text{offset}}$):** 가중치 마스크를 적용한 L1 손실이다.
$$L_{\text{offset}} = \frac{1}{|P_{\text{pseudo}}|} \sum_{(i,j) \in P_{\text{pseudo}}} |O(i,j) - \hat{O}(i,j)| + \frac{1}{|P_{\text{refined}}|} \sum_{(i,j) \in P_{\text{refined}}} W(i,j) \cdot |O(i,j) - \bar{O}(i,j)|$$
- **시맨틱 맵 손실 ($L_{\text{sem}}$):** 일반적인 교차 엔트로피(Cross-Entropy) 손실을 사용한다.
$$L_{\text{sem}} = -\frac{1}{|P_{\text{sem}}|} \sum_{(i,j) \in P_{\text{sem}}} \log S(i,j)$$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** PASCAL VOC 2012, MS COCO 2017.
- **지표:** $\text{mAP}_{50}$ (VOC), $\text{AP}$ (COCO).
- **백본 네트워크:** HRNet-48.
- **WSSS 방법:** saliency map을 사용하지 않는 PMM 채택.

### 2. 주요 결과

- **성능:** 기성 제안 기술 없이도 VOC 2012에서 $\text{mAP}_{50}$ 51.0%, COCO에서 $\text{AP}_{50}$ 28.0%를 달성하였다. 이는 외부 제안 기술을 사용한 기존 방법들과 대등하거나 이를 상회하는 수준이다.
- **포인트 지도 학습:** PAM의 힌트 대신 포인트 레이블을 사용했을 때, VOC 2012에서 $\text{mAP}_{50}$ 56.1%(MRCNN 정제 전)까지 성능이 향상되어 포인트 레이블의 효율성을 입증하였다.

### 3. 절제 연구 (Ablation Study)

- **PAM의 효과:** 일반 CAM 대비 true-positive 샘플 수를 3배 이상 증가시켰으며, $\text{mAP}_{50}$를 16.4% 향상시켰다.
- **IAG의 효과:** IAG가 없을 경우 시맨틱 드리프트 문제로 인해 성능이 9.9% 하락하며 로컬 미니멈에 빠지는 경향을 보였다.
- **Self-Refinement의 효과:** 정제 과정을 통해 가짜 레이블의 품질이 점진적으로 향상되며 성능이 2.3% 추가 상승하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 WSIS 분야에서 외부 제안 기술에 대한 의존성을 완전히 제거함으로써, 모델의 도메인 범용성을 높이고 엄격한 이미지 레벨 지도 학습 설정을 구현했다는 점에서 큰 의의가 있다. 특히, '시맨틱 드리프트'라는 문제를 명시적으로 정의하고 IAG라는 단순하지만 효과적인 방법으로 이를 해결한 점이 돋보인다.

### 한계 및 논의

- **객체 중첩 문제:** 논문에서 언급되었듯이, 이미지 내 객체들이 서로 겹쳐 있을 경우(overlapping) 가짜 레이블 생성 단계에서 true-positive를 확보하는 데 한계가 있다. 이는 COCO 데이터셋의 성능이 VOC보다 상대적으로 낮게 나오는 원인이 된다.
- **WSSS 의존성:** WSIS 성능이 사용된 WSSS 방법의 품질에 어느 정도 영향을 받는다. 다만, 본 연구는 WSSS의 성능 변화에 대해 비교적 강건(robust)함을 보였으며, 향후 WSSS 기술의 발전이 곧 BESTIE의 성능 향상으로 이어질 가능성이 크다.

## 📌 TL;DR

BESTIE는 외부 제안 기술 없이 오직 이미지 레벨 레이블만을 사용하여 인스턴스 분할을 수행하는 프레임워크이다. WSSS의 지식을 전이하고 PAM을 통해 정밀한 인스턴스 힌트를 추출하며, 시맨틱 드리프트를 해결하는 IAG와 온라인 Self-Refinement 과정을 통해 가짜 레이블의 품질을 높인다. 이 연구는 WSIS의 학습 설정 정의를 바로잡고 성능을 크게 향상시켰으며, 향후 포인트 지도 학습 등으로 확장 가능한 유연한 구조를 제시한다.
