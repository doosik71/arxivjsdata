# Tracking Every Thing in the Wild

Siyuan Li, Martin Danelljan, Henghui Ding, Thomas E. Huang, Fisher Yu (2022)

## 🧩 Problem to Solve

본 논문은 다중 카테고리 다중 객체 추적(Multi-category Multiple Object Tracking, MOT) 시스템이 가진 근본적인 설계 및 평가 방식의 문제를 해결하고자 한다.

기존의 MOT 방법론과 평가지표(Metric)들은 분류(Classification) 성능이 거의 완벽하다는 가정을 전제로 한다. 즉, 객체를 먼저 탐지하고 분류한 뒤, 동일한 클래스로 예측된 객체들 사이에서만 연관성(Association)을 찾는다. 평가지표 역시 클래스 라벨을 기준으로 결과를 그룹화하여 클래스별로 성능을 측정한다.

그러나 실제 대규모 데이터셋(예: TAO, BDD100K)은 다음과 같은 특성으로 인해 분류 성능이 낮을 수밖에 없다:
1. **Long-tailed Distribution**: 소수의 빈번한 클래스가 지배적이며, 수많은 희귀 클래스가 존재한다.
2. **Semantic Similarity**: 버스와 밴처럼 의미적으로 매우 유사한 세분화된(Fine-grained) 클래스들이 존재하여 오분류가 빈번하다.
3. **Incomplete Annotation**: 대규모 데이터셋의 경우 모든 객체를 완벽하게 라벨링하는 비용이 너무 커서 일부만 라벨링 된 경우가 많다.

결과적으로, 분류 성능이 조금만 떨어져도 객체의 위치 추적(Localization)과 연관성(Association) 성능이 완벽함에도 불구하고 전체 추적 성능이 0점으로 처리되는 불합리한 상황이 발생한다. 따라서 본 논문은 **분류와 추적을 분리(Disentangle)** 하여, 분류가 부정확한 상황에서도 객체의 궤적을 효과적으로 추적하고 이를 공정하게 평가하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 분류(Classification)를 추적(Tracking)의 전제 조건이 아닌, 독립적인 요소로 취급하는 것이다. 이를 위해 다음과 같은 두 가지 핵심 기여를 제시한다.

1. **TETA (Track Every Thing Accuracy) Metric**: 위치 기반의 Local Cluster Evaluation을 도입하여, 예측된 클래스 라벨이 틀렸더라도 위치와 연관성 성능을 독립적으로 측정할 수 있는 새로운 평가지표를 제안한다. 또한, 불완전한 라벨링(Incomplete annotation) 문제로 인해 발생하는 잘못된 FP(False Positive) 판정 문제를 해결한다.
2. **TETer (Track Every Thing tracker)**: "모든 것을 연관시킨다"는 Associate-Every-Thing (AET) 전략을 기반으로 한 추적기를 제안한다. 특히, 하드한 클래스 라벨 대신 소프트한 클래스 정보인 **Class Exemplar Matching (CEM)**을 도입하여, 분류 오류에 강건하면서도 클래스의 세부 정보를 활용한 연관성을 수행한다.

## 📎 Related Works

기존의 MOT는 주로 **Tracking-by-Detection** 패러다임을 따른다. 이는 탐지기(Detector)가 생성한 바운딩 박스를 기반으로 외형 특징(Appearance features)이나 모션 모델을 이용해 동일 객체를 연결하는 방식이다.

최근 Open-set MOT 연구들이 등장하며 카테고리에 관계없이 모든 객체를 추적하려는 시도가 있었으나, 실제 응용 분야(예: 비디오 분석)에서는 여전히 분류 정보가 필수적이다. 하지만 기존의 다중 카테고리 MOT 방식들은 각 클래스를 독립적으로 처리함으로써, 앞서 언급한 Long-tailed 분포와 분류 오류 문제에 매우 취약하다.

평가지표 측면에서 MOTA, IDF1, HOTA 등은 단일 카테고리 추적을 위해 설계되었다. 이를 다중 카테고리로 확장할 때 단순히 클래스별로 그룹화하여 평균을 내는 방식을 사용하는데, 이는 분류 성능이 낮을 때 추적 성능을 과소평가하는 경향이 있으며, TAO와 같은 불완전 라벨링 데이터셋에서는 잘못된 FP를 생성하도록 유도하는 등 'Game-able'한 특성을 보인다.

## 🛠️ Methodology

### 1. TETA Metric (Track Every Thing Accuracy)

TETA는 HOTA를 확장하여 Localization, Association, Classification 세 가지 점수를 독립적으로 산출하고 그 산술 평균을 구한다.

#### Local Cluster Evaluation
불완전한 라벨링 상황에서 잘못된 FP 판정을 막기 위해 도입되었다. 각 Ground Truth(GT) 박스를 클러스터의 중심(Anchor)으로 설정하고, IoU 마진 $r$ 내에 있는 예측 박스들만 평가 대상에 포함시킨다. 이 범위 밖에 있는 예측은 무시함으로써, 라벨링이 누락된 객체를 추적했을 때 발생하는 억울한 FP 페널티를 방지한다. 또한, 클래스 라벨이 아닌 **위치**를 기준으로 그룹화를 수행하므로, 분류가 틀려도 추적 성능을 측정할 수 있다.

#### 세부 점수 산출 방식
- **Localization Score ($\text{LocA}$)**: 헝가리안 알고리즘을 통해 최적 매칭을 찾고, Jaccard Index를 사용하여 계산한다.
$$\text{LocA} = \frac{|\text{TPL}|}{|\text{TPL}| + |\text{FPL}| + |\text{FNL}|}$$
- **Association Score ($\text{AssocA}$)**: 각 매칭된 객체($b \in \text{TPL}$)에 대해 HOTA의 정의를 따라 연관성 점수를 구하고, 이를 모든 $\text{TPL}$에 대해 평균 낸다.
$$\text{AssocA}(b) = \frac{|\text{TPA}(b)|}{|\text{TPA}(b)| + |\text{FPA}(b)| + |\text{FNA}(b)|}, \quad \text{AssocA} = \frac{1}{|\text{TPL}|} \sum_{b \in \text{TPL}} \text{AssocA}(b)$$
- **Classification Score ($\text{ClsA}$)**: 위치 매칭이 성공한($\text{IoU} \ge 0.5$) 객체들에 대해서만 실제 클래스와 예측 클래스가 일치하는지를 독립적으로 측정한다.
$$\text{ClsA} = \frac{|\text{TPC}|}{|\text{TPC}| + |\text{FPC}| + |\text{FNC}|}$$
- **최종 점수**: 세 점수의 산술 평균을 사용한다.
$$\text{TETA} = \frac{\text{LocA} + \text{AssocA} + \text{ClsA}}{3}$$

### 2. TETer Tracker

#### Class-Agnostic Localization (CAL)
희귀 클래스의 경우 분류기 성능이 낮아 NMS 과정에서 제거될 가능성이 높다. 이를 방지하기 위해 클래스 확률을 무시하고 모든 객체를 탐지하는 **Class-agnostic NMS**를 사용하여 최대한 많은 객체를 후보군으로 확보한다.

#### Class Exemplar Matching (CEM)
분류기의 하드한 예측값 대신, 대조 학습(Contrastive Learning)을 통해 학습된 클래스별 '대표 특징(Exemplar)'을 사용한다.

**U-SupCon Loss**: 학습 시 양성 샘플의 수가 가변적인 문제를 해결하기 위해 Unbiased Supervised Contrastive Loss를 제안한다.
$$\mathcal{L}_C = -\sum_{q \in Q} \frac{1}{|Q^+(q)|} \sum_{q^+ \in Q^+(q)} \log \frac{\exp(\text{sim}(q, q^+) / \tau)}{\text{PosD}(q) + \sum_{q^- \in Q^-(q)} \exp(\text{sim}(q, q^-) / \tau)}$$
여기서 $\text{PosD}(q)$는 다음과 같이 정의되어 손실 함수의 하한선 변동을 방지한다.
$$\text{PosD}(q) = \frac{1}{|Q^+(q)|} \sum_{q^+ \in Q^+(q)} \exp(\text{sim}(q, q^+) / \tau)$$

#### Association 및 Correction 절차
1. **후보군 선정**: 쿼리 객체의 클래스 엠플러(Class Exemplar)와 다음 프레임 객체들의 엠플러 간 코사인 유사도를 계산하여 유사도가 $\delta > 0.5$인 후보군 $\mathcal{C}$를 선정한다. 이는 클래스 정보를 소프트한 우선순위(Soft prior)로 활용하는 것이다.
2. **인스턴스 연관성**: 선정된 후보군 내에서 인스턴스 수준의 특징(Instance features)과 양방향 소프트맥스(Bidirectional softmax)를 이용해 최종 매칭을 결정한다.
3. **Temporal Class Correction (TCC)**: 추적된 궤적(Track) 전체에서 예측된 클래스들의 다수결(Majority vote)을 통해 각 프레임의 분류 오류를 보정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: TAO (800개 이상의 클래스, 불완전 라벨링), BDD100K (주행 영상, Long-tailed 분포).
- **비교 대상**: SORT, DeepSORT, Tracktor++, QDTrack, AOA 등.
- **지표**: TETA ($\text{LocA}, \text{AssocA}, \text{ClsA}$), mMOTA, mIDF1.

### 주요 결과
1. **Metric 분석**: 
   - **Cross Dataset Consistency**: TAO 지표로 최적화된 모델은 BDD100K에서 성능이 급락했으나, TETA로 최적화된 모델은 일관된 성능을 보였다.
   - **Robustness**: 단순한 Copy & Paste 기법으로 TAO 지표는 크게 상승(치팅 가능)했지만, TETA는 오히려 하락하여 훨씬 엄격하고 정확한 평가가 가능함을 입증했다.
2. **추적 성능 (TETer)**:
   - **TAO**: TETA 기준 33.25점으로 기존 SOTA인 QDTrack(30.00)을 상회했으며, 특히 $\text{AssocA}$에서 7점 이상의 큰 향상을 보였다.
   - **BDD100K**: mMOTA(39.1), mIDF1(53.3) 모두에서 SOTA를 달성했다.
3. **CEM의 범용성**: CEM 모듈을 DeepSORT, Tracktor++, QDTrack에 각각 적용했을 때, 모든 모델에서 $\text{AssocA}$ 점수가 일관되게 상승하는 것을 확인하여 CEM이 독립적인 플러그인 모듈로서 가치가 있음을 보였다.

## 🧠 Insights & Discussion

본 논문은 MOT 시스템에서 분류 성능에 대한 과도한 의존성이 실제 환경에서의 추적 성능을 저해하고, 이를 제대로 평가하지 못하게 만든다는 점을 날카롭게 지적했다. 

**강점**:
- **분류-추적 분리**: 분류 성능이 낮더라도 궤적을 유지할 수 있는 AET 전략과 이를 측정할 수 있는 TETA 지표를 동시에 제시하여 논리적 완결성을 갖추었다.
- **CEM의 효율성**: 하드 라벨 대신 임베딩 기반의 유사도를 사용하여 세분화된 클래스 간의 관계를 유연하게 처리했다.

**한계 및 논의**:
- **탐지기 의존성**: CAL을 사용하더라도 근본적으로는 탐지기(Detector)가 객체를 찾아내야만 추적이 가능하다. 탐지 자체가 실패한 경우(예: 극심한 가림 현상)에 대한 대책은 부족하다.
- **실패 사례**: 정성적 분석 결과, 객체의 외형 변화가 매우 급격하거나(TAO 데이터셋 특성), 가림(Occlusion)이 심한 경우 여전히 추적 실패가 발생한다.
- **가정**: TETA의 Local Cluster Evaluation에서 마진 $r$의 설정값이 성능 측정에 영향을 줄 수 있으며, 데이터셋의 밀집도에 따라 적절한 $r$을 찾는 과정이 필요하다.

## 📌 TL;DR

본 논문은 대규모 Long-tailed MOT 환경에서 분류 오류가 추적 성능 평가와 실제 추적에 미치는 악영향을 해결하기 위해 **TETA 평가지표**와 **TETer 추적기**를 제안한다. TETA는 위치 기반 클러스터링을 통해 분류와 관계없이 추적 성능을 측정하며, TETer는 클래스 엠플러 매칭(CEM)을 통해 분류 오류에 강건한 연관성을 수행한다. 이 연구는 향후 복잡한 실제 환경에서 수많은 카테고리의 객체를 안정적으로 추적하고 공정하게 평가하는 표준을 제시할 가능성이 높다.