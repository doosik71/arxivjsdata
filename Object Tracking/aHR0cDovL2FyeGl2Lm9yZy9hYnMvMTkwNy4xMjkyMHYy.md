# Tracking Holistic Object Representations

Axel Sauer, Elie Aljalbout, Sami Haddadin (2019)

## 🧩 Problem to Solve

본 논문은 비주얼 트래킹(Visual Tracking)에서 대상 객체의 외형 변화(Appearance Change)로 인해 발생하는 성능 저하 문제를 해결하고자 한다. 특히 Siamese 네트워크 기반의 템플릿 매칭(Template Matching) 방식은 초기 프레임의 템플릿에 의존하는 경향이 있어, 추적 과정 중 발생하는 회전, 조명 변화, 폐쇄(Occlusion), 모션 블러 및 형태 변형과 같은 동적인 환경 변화에 취약하다는 한계가 있다.

연구의 목표는 기존의 Siamese 트래커를 그대로 유지하면서, 추가적인 네트워크 재학습 없이도 객체의 다양한 상태를 포괄할 수 있는 '전체론적 객체 표현(Holistic Object Representation)'을 구축하는 프레임워크를 제안하는 것이다. 이를 통해 트래커의 강건성(Robustness)과 정확도를 동시에 향상시키는 것을 목적으로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 추적 과정 중에 객체의 다양한 외형을 대표하는 템플릿들을 수집하여 템플릿 모듈을 구성하는 것이다. 단순히 많은 템플릿을 저장하는 것이 아니라, 특성 공간(Feature Space) 내에서 서로 가장 멀리 떨어져 있는, 즉 **다양성(Diversity)이 극대화된 템플릿들만을 선별적으로 유지**함으로써 효율적인 객체 표현을 구축한다.

주요 기여 사항은 다음과 같다.
- Siamese 특성 공간에서의 다양성을 측정하기 위해 Gram 행렬의 행렬식(Determinant)을 이용한 분석적 방법론을 제안하였다.
- 장기 기억을 담당하는 Long-term Module(LTM)과 단기 변화를 처리하는 Short-term Module(STM)로 구성된 계층적 구조를 설계하였다.
- 기존의 어떤 템플릿 매칭 기반 트래커에도 즉시 적용 가능한 플러그인 형태의 프레임워크를 제시하였으며, 이는 추가 학습 없이도 성능을 향상시킨다.

## 📎 Related Works

기존의 템플릿 매칭 방식은 이미지 강도, 그래디언트 특성 또는 DNN 기반의 특성 추출기를 사용하여 타겟과 후보 영역 간의 유사도를 측정한다. 최근에는 SiamFC, SiamRPN과 같은 Siamese 네트워크가 주류를 이루고 있으며, 이들은 특성 임베딩 공간에서의 상관관계 연산을 통해 실시간 성능을 확보하였다.

객체 외형 변화를 해결하기 위한 기존의 다중 템플릿(Multi-template) 접근 방식으로는 다음과 같은 연구들이 있다.
- **MemTrack**: LSTM 기반의 메모리 컨트롤러를 사용하여 템플릿을 읽고 쓰는 동적 메모리 네트워크를 사용한다. 하지만 이는 메모리 네트워크와 트래커를 동시에 학습시켜야 한다는 제약이 있다.
- **MMLT**: 여러 특성 템플릿을 축적한 뒤 이를 가중 결합하여 컨볼루션을 수행한다. 하지만 저자들은 저장된 모든 템플릿이 실제 타겟 객체라는 가정하에만 수학적 결합이 유효하며, 드리프트(Drift) 발생 시 위험할 수 있다고 지적한다.

본 논문의 THOR는 학습 기반이 아닌 분석적(Analytical) 방법을 사용하여 해석 가능성을 높였으며, 템플릿을 결합하는 대신 개별적으로 유지함으로써 드리프트에 더 강건하게 대응한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

THOR 프레임워크는 Long-term Module(LTM)과 Short-term Module(STM)이라는 두 개의 핵심 모듈과 이를 제어하는 추론 전략으로 구성된다.

### 1. Long-term Module (LTM)
LTM의 목적은 객체의 다양한 상태(조명, 형태 등)를 대표하는 템플릿을 저장하여 장기적인 추적 및 재검출을 가능하게 하는 것이다.

**다양성 측정 및 할당 전략**:
특성 벡터 $f_1, \dots, f_n$들이 이루는 평행육면체(Parallelotope)의 부피 $\Gamma$를 최대화하는 것이 목표이다. 이를 위해 Gram 행렬 $G$를 다음과 같이 정의한다.
$$G(f_1, \dots, f_n) = \begin{bmatrix} f_1 \cdot f_1 & f_1 \cdot f_2 & \dots & f_1 \cdot f_n \\ f_2 \cdot f_1 & f_2 \cdot f_2 & \dots & f_2 \cdot f_n \\ \vdots & \vdots & \ddots & \vdots \\ f_n \cdot f_1 & f_n \cdot f_2 & \dots & f_n \cdot f_n \end{bmatrix}$$
이 Gram 행렬의 행렬식 $|G|$는 부피 $\Gamma$의 제곱에 비례하므로, 목적 함수는 다음과 같다.
$$\max_{f_1, \dots, f_n} \Gamma(f_1, \dots, f_n) \propto \max_{f_1, \dots, f_n} |G(f_1, \dots, f_n)|$$
새로운 템플릿 후보가 기존 템플릿 중 하나를 대체했을 때 $|G|$를 증가시킨다면, 해당 템플릿을 메모리에 저장한다.

**드리프트 방지를 위한 하한선(Lower Bound)**:
잘못된 템플릿이 저장되는 것을 막기 위해 기본 템플릿 $T_1$과의 유사도 하한선을 둔다.
- **Dynamic Lower Bound**: STM에서 제공하는 다양성 지표 $\gamma$를 반영하여 $f_c \cdot f_1 > \ell \cdot G_{11} - \gamma$ 조건을 만족해야 한다.
- **Ensemble Lower Bound**: 모든 LTM 템플릿과의 유사도를 체크하여 $\inf f_c \cdot f_{1:n} > \ell \cdot \text{diag}(G)$ 조건을 만족해야 한다.

### 2. Short-term Module (STM)
STM은 급격한 움직임이나 부분 폐쇄와 같이 LTM이 처리하기 어려운 단기적 변화를 담당한다. FIFO(First-In-First-Out) 방식으로 고정된 개수 $K_{st}$만큼의 템플릿을 유지한다.

STM은 또한 다양성 지표 $\gamma$를 계산하여 LTM에 전달한다. $\gamma$는 Gram 행렬 $G_{st}$의 상삼각 성분의 합을 최대값으로 정규화하여 계산한다.
$$\gamma = 1 - \frac{2}{N(N+1)G_{st, \max}} \sum_{i<j} G_{st, ij}$$
$\gamma$가 1에 가까울수록 STM 내의 템플릿들이 더 다양함을 의미한다.

### 3. 추론 전략 (Inference Strategy)
- **Modulation**: 모든 템플릿의 Activation Map을 생성한 뒤, 각 템플릿의 최대 점수를 가중치로 하여 공간적 평균을 구하고 이를 다시 모든 맵에 곱해 정규화함으로써 정보의 통합을 꾀한다.
- **ST-LT Switch**: 기본적으로는 STM의 예측값을 사용한다. 하지만 STM의 예측값과 LTM의 예측값 사이의 $IoU$가 임계값 $th_{iou}$보다 낮을 경우, 더 강건한 LTM의 예측값을 선택하고 STM을 재초기화한다.

## 📊 Results

### 실험 설정
- **비교 대상**: SiamFC, SiamRPN, SiamMask 등 대표적인 Siamese 트래커.
- **데이터셋**: VOT2018, OTB2015.
- **지표**: EAO (Expected Average Overlap), Accuracy, Robustness, AUC, Precision.

### 주요 결과
1. **정량적 성능 향상**: VOT2018 벤치마크에서 THOR를 적용한 모든 트래커의 EAO가 향상되었다. 특히 `THOR-SiamRPN`은 더 복잡한 구조인 SiamRPN++의 성능(EAO 0.414)에 근접하는 0.416을 달성하였다.
2. **강건성 및 정밀도**: OTB2015에서 AUC와 Precision 모두 유의미하게 상승하였으며, 이는 THOR가 드리프트를 효과적으로 억제하고 객체를 더 정확하게 추적함을 보여준다.
3. **속도 분석**: 다중 템플릿 연산으로 인해 속도는 약간 감소하지만, 여전히 실시간 수준을 유지한다. `THOR-SiamRPN`의 경우 244 FPS라는 매우 빠른 속도로 동작한다.
4. **Proof of Concept**: 추적 과정 중 Gram 행렬식 $|G|$가 점진적으로 증가하다가 수렴하는 것을 확인하였으며, 이는 객체에 대한 정보가 포화 상태에 이를 때까지 다양성이 확장됨을 입증한다.

### Ablation Study
- Modulation과 STM을 제거했을 때 성능 하락이 가장 컸으며, 이는 단기/장기 메모리의 조화와 정보 통합 과정이 필수적임을 시사한다.
- Ensemble 하한선 전략이 Dynamic 전략보다 더 높은 다양성($|G|_{norm}$)을 확보하면서도 드리프트를 낮게 유지하는 것으로 나타났다.

## 🧠 Insights & Discussion

본 논문은 Siamese 트래커의 특성 공간을 기하학적으로 해석하여, '부피'라는 개념을 통해 템플릿의 다양성을 정량화했다는 점에서 매우 창의적이다. 특히 추가적인 학습 없이 기존 모델에 얹어서 사용할 수 있는 플러그인 구조라는 점이 실용적인 강점으로 작용한다.

**한계 및 논의점**:
- **시퀀스 길이의 영향**: 짧은 시퀀스에서는 템플릿을 충분히 축적할 시간이 부족하여 성능 향상 폭이 적다. 이는 LTM의 특성상 충분한 샘플링 시간이 필요하기 때문이다.
- **하이퍼파라미터 민감도**: 하한선 $\ell$이나 $IoU$ 임계값 등에 따라 성능이 좌우될 수 있으며, 이는 기반이 되는 Siamese 트래커 자체의 민감도 문제와 맞물려 있다.
- **장기 추적(Long-term Tracking)**: 본 연구는 단기 벤치마크 위주로 실험되었으나, 저자들은 LTM의 특성상 OxUvA와 같은 장기 추적 데이터셋에서 더 큰 효과를 낼 것으로 기대하고 있다.

## 📌 TL;DR

본 논문은 Siamese 트래커의 성능을 높이기 위해 Gram 행렬식을 이용한 다양성 기반의 다중 템플릿 관리 프레임워크인 **THOR**를 제안한다. LTM(장기)과 STM(단기) 모듈을 통해 객체의 전체론적 표현을 구축하며, 추가 학습 없이도 기존 트래커의 강건성과 정확도를 대폭 향상시킨다. 특히 분석적인 다양성 측정 방식을 통해 연산 효율성을 확보하여 실시간성을 유지하면서도 최신 SOTA 모델에 근접하는 성능을 보여준다는 점에서 가치가 높다.