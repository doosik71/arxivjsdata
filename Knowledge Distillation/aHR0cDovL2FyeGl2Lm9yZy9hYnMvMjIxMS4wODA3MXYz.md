# KD-DETR: Knowledge Distillation for Detection Transformer with Consistent Distillation Points Sampling

Yu Wang, Xin Li, Shengzhao Weng, Gang Zhang, Haixiao Yue, Haocheng Feng, Junyu Han, Errui Ding (2025)

## 🧩 Problem to Solve

본 논문은 DETR(Detection Transformer) 계열의 객체 검출 모델을 압축하기 위한 Knowledge Distillation(KD) 방법론을 다룬다. DETR은 기존의 CNN 기반 검출기에 비해 뛰어난 성능을 보이지만, 모델 규모가 커짐에 따라 계산 비용이 증가하여 실시간 서비스 배포에 어려움이 있다.

기존 CNN 기반 검출기의 KD는 고정된 앵커(Anchor)나 슬라이딩 윈도우 방식을 사용하므로, 교사(Teacher) 모델과 학생(Student) 모델 사이에 공간적 대응 관계가 명확한 '일관된 증류 지점(Consistent Distillation Points)'이 존재한다. 반면, DETR은 학습 가능한 Object Query를 통해 집합 예측(Set Prediction) 방식을 취한다. 각 모델의 Object Query는 독립적으로 최적화되므로 서로 다른 특징에 집중하는 '자기중심적(Egocentric)' 특성을 가진다. 이로 인해 교사와 학생 모델 간의 증류 지점이 일치하지 않는 '불일치 문제(Inconsistency)'가 발생하며, 이는 효과적인 지식 전달을 방해하는 핵심 원인이 된다.

따라서 본 논문의 목표는 DETR 구조에서도 교사와 학생 모델 간에 충분하고 일관된 증류 지점을 확보하여, 모델 압축 시 성능 저하를 최소화하고 정확도를 높이는 일반적인 KD 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **검출 작업(Detection Task)과 증류 작업(Distillation Task)을 분리(Decouple)**하는 것이다. 이를 위해 교사와 학생 모델이 공유하는 별도의 '전용 객체 쿼리(Specialized Object Queries)' 세트를 도입하여 일관된 증류 지점을 강제로 생성한다.

주요 기여 사항은 다음과 같다.

1. **KD-DETR 프레임워크 제안**: 동종(Homogeneous) 및 이종(Heterogeneous) 증류 모두에 적용 가능한 일반적인 DETR 지식 증류 패러다임을 제시하였다.
2. **일관된 증류 지점 샘플링 전략**: 일반적 샘플링(General Sampling)과 구체적 샘플링(Specific Sampling)을 결합하여 교사 모델의 광범위하고 정밀한 지식을 모두 추출하는 전략을 제안하였다.
3. **전경 재균형 가중치(Foreground Rebalance Weight) 도입**: 교사 모델의 예측 확률을 이용하여 전경(Foreground) 지역의 증류에 더 높은 가중치를 부여함으로써 배경 노이즈 문제를 해결하였다.
4. **이종 증류로의 확장**: CNN 검출기의 앵커 좌표를 DETR의 쿼리 형태로 변환하여, 서로 다른 아키텍처 간의 지식 전이가 가능함을 입증하였다.

## 📎 Related Works

### 기존 객체 검출 방식

- **CNN 기반 검출기**: Faster R-CNN과 같은 2단계 검출기나 YOLO, FCOS와 같은 1단계 검출기는 앵커 기반의 검증(Verification) 문제로 접근한다. 이들은 고정된 공간적 우선순위를 가지므로 KD 적용 시 대응 지점을 찾기가 매우 쉽다.
- **DETR 기반 검출기**: DETR은 Bipartite Matching을 이용한 집합 예측 문제로 접근하며, Object Query를 통해 객체를 탐색한다. Deformable DETR, DAB-DETR, DINO 등으로 발전하며 성능이 향상되었으나, 모델 규모가 크다는 단점이 있다.

### 기존 Knowledge Distillation (KD)

- **일반적 KD**: 로짓(Logits), 중간 특징 맵(Intermediate Activations), 특징 간의 관계(Relation)를 모방하는 방식으로 나뉜다.
- **DETR KD의 한계**: 기존 연구(예: DETRDistill)는 Hungarian Matching을 통해 쿼리 간의 대응 관계를 찾으려 했으나, 매칭 과정이 불안정하고 단순한 유사성만으로 일관성을 확보하기에는 부족함이 있었다.

## 🛠️ Methodology

### 전체 시스템 구조

KD-DETR은 학습 과정에서 검출을 위한 원래의 쿼리 $\text{q}$와 증류를 위한 전용 쿼리 $\tilde{\text{q}}$를 분리하여 사용한다.

1. **검출 작업**: 학생 모델은 원래의 입력 $\text{x} = \{I, \text{q}\}$를 사용하여 Ground Truth와 매칭하고 표준 검출 손실 $\mathcal{L}_{\text{det}}$을 계산한다.
2. **증류 작업**: 공유된 증류 지점 $\tilde{\text{x}} = \{I, \tilde{\text{q}}\}$를 교사와 학생 모델에 동시에 입력하여 각각의 예측값 $\text{c}, \text{b}$ (분류 및 박스 위치)를 얻고, 그 차이를 최소화하는 $\mathcal{L}_{\text{distill}}$을 계산한다.

### 손실 함수 및 학습 절차

최종 손실 함수는 다음과 같이 정의된다:
$$\mathcal{L} = \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{det}}$$

증류 손실 $\mathcal{L}_{\text{distill}}$은 전경 재균형 가중치 $w_i$를 적용하여 다음과 같이 계산된다:
$$\mathcal{L}_{\text{distill}} = \sum_{i=1}^{M} w_i \left[ \lambda_{\text{cls}} \mathcal{L}_{\text{KL}}(\hat{c}_t^i \| \hat{c}_s^i) + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}}(b_s^i, b_t^i) + \lambda_{\text{GIoU}} \mathcal{L}_{\text{GIoU}}(b_s^i, b_t^i) \right]$$

여기서 $\mathcal{L}_{\text{KL}}$은 분류 로짓에 대한 KL-Divergence이며, $\mathcal{L}_{\text{L1}}$과 $\mathcal{L}_{\text{GIoU}}$는 박스 위치 회귀를 위한 손실 함수이다. $w_i$는 교사 모델이 예측한 클래스 확률의 최댓값으로, 전경 객체일수록 더 큰 가중치를 갖게 된다:
$$w_i = \max_{c \in [0, K]} p_t(y_c | q_i)$$

### 증류 지점 샘플링 전략 $\tilde{\text{q}} = \{\text{q}_g, \text{q}_s\}$

교사 모델의 지식을 포괄적으로 추출하기 위해 두 가지 전략을 사용한다.

- **일반적 샘플링(General Sampling, $\text{q}_g$)**: 전체 특징 맵을 균등하게 스캔하기 위해 랜덤하게 초기화된 쿼리를 사용한다. 이 쿼리들은 학습되지 않는 고정값이며 매 반복(Iteration)마다 다시 샘플링되어 교사의 전반적인 응답을 탐색한다.
- **구체적 샘플링(Specific Sampling, $\text{q}_s$)**: 교사 모델에서 이미 잘 최적화된 Object Query를 그대로 가져와 사용한다 ($\text{q}_s = \text{q}_{\text{teacher}}$). 이는 교사가 특히 주목하는 정밀한 영역의 지식을 전달하기 위함이다.

### 이종 증류(Heterogeneous Distillation) 확장

DETR(교사) $\rightarrow$ CNN(학생) 구조로 지식을 전달하기 위해, CNN의 앵커 좌표 $\text{A} = \{x_a, y_a, w_a, h_a\}$를 DETR의 Object Query 형태로 변환하는 방식을 제안한다.
$$\text{Q}_A = \text{MLP}(\text{PE}(x_a + \frac{w_a}{2}, y_a + \frac{h_a}{2}, w_a, h_a))$$
이렇게 변환된 쿼리를 통해 아키텍처가 다른 두 모델 간에도 공간적 일관성을 유지하며 지식을 증류할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: MS COCO 2017
- **대상 모델**: DAB-DETR, Deformable DETR, DINO
- **백본**: ResNet-18 (학생), ResNet-50/101 (교사)
- **지표**: mAP, $\text{AP}_{50}$, $\text{AP}_{75}$

### 주요 결과

1. **동종 증류 성능 향상**:
   - DAB-DETR (ResNet-18 학생): Baseline 대비 **5.2% mAP 향상** (36.2% $\rightarrow$ 41.4%).
   - Deformable DETR (ResNet-18 학생): Baseline 대비 **3.6% mAP 향상** (40.1% $\rightarrow$ 43.7%).
   - DINO (ResNet-18 학생): Baseline 대비 **4.4% mAP 향상** (44.0% $\rightarrow$ 48.4%).
   - 특히 ResNet-50 학생 모델의 경우, 증류를 통해 교사 모델의 성능을 일부 상회하는 결과가 나타났다.

2. **이종 증류 성능**:
   - 교사(DINO ResNet-50) $\rightarrow$ 학생(Faster R-CNN ResNet-50) 증류 시 **2.1% mAP 향상**을 달성하였으며, 이는 동종 증류 방법들과 경쟁 가능한 수준이다.

3. **SOTA 비교**: Deformable DETR 기반 실험에서 기존의 DETRDistill보다 **1.7% 더 높은 성능**을 보이며 우수성을 입증하였다.

## 🧠 Insights & Discussion

### 분석 및 강점

- **쿼리 수 증가와의 차별점**: 단순히 학생 모델의 Object Query 수를 늘리는 것만으로는 성능 향상이 미미했다(Table 5). 이는 성능 향상의 주원인이 쿼리의 양적 증가가 아니라, 교사 모델로부터 전이된 '지식'에 있음을 시사한다.
- **샘플링 전략의 상호보완성**: 시각화 결과, Specific Sampling은 전경 객체에 집중하는 반면, General Sampling은 배경 및 GT에 없는 객체 등 추가적인 세만틱 정보를 제공한다. 이 두 전략을 결합했을 때 가장 높은 성능 향상이 나타났다.
- **증류 지점 밀도의 트레이드-오프**: 일반 샘플링 쿼리를 10개에서 300개까지 늘릴 때는 성능이 상승하지만, 900개까지 과도하게 늘리면 오히려 배경 노이즈가 증가하여 성능이 소폭 하락하는 경향을 보였다.

### 한계 및 논의사항

- 본 연구는 주로 로짓 기반의 증류에 집중하고 있으며, 특징 맵 수준의 깊은 특징 증류(Feature-based distillation)와의 결합 가능성에 대해서는 명확히 다루지 않았다.
- 이종 증류 시 앵커를 쿼리로 변환하는 MLP 과정에서의 최적화 방식이 결과에 미치는 영향에 대한 추가 분석이 필요할 수 있다.

## 📌 TL;DR

본 논문은 DETR 계열 모델의 지식 증류 시 발생하는 **'증류 지점의 불일치'** 문제를 해결하기 위해, 검출과 증류 작업을 분리하고 교사-학생이 공유하는 **전용 증류 쿼리**를 도입한 **KD-DETR**을 제안한다. 랜덤 샘플링과 교사 쿼리 재사용을 결합한 전략 및 전경 가중치 조절을 통해, 다양한 DETR 모델에서 2.6%~5.2%의 mAP 향상을 이루었으며, 특히 서로 다른 구조(DETR $\rightarrow$ CNN) 간의 지식 전이 가능성을 입증하였다. 이는 향후 고성능 DETR 모델을 경량화하여 실제 환경에 배포하는 데 중요한 기반 기술이 될 것으로 보인다.
