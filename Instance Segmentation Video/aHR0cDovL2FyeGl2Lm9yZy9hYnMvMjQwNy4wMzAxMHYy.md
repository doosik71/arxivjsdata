# CAVIS: Context-Aware Video Instance Segmentation

Seunghun Lee, Jiwan Seo, Kiljoon Han, Minwoo Choi, and Sunghoon Im (2024)

## 🧩 Problem to Solve

본 논문은 Video Instance Segmentation (VIS) 분야에서 발생하는 객체 식별 및 추적의 불안정성 문제를 해결하고자 한다. VIS는 비디오 시퀀스 내에서 개별 객체를 분할하고 동일성을 유지하며 식별하는 작업으로, 자율 주행 및 비디오 편집 등 다양한 분야에서 핵심적인 역할을 한다.

최근의 query-based segmentation 아키텍처들은 객체의 중심점(center)이나 특징(feature) 간의 유사성을 기반으로 프레임 간 객체를 연결하는 방식을 사용한다. 그러나 이러한 방식은 객체가 다른 객체에 의해 심하게 가려지는 Occlusion 상황이나, 외형이 매우 유사한 여러 객체가 동시에 등장하는 시나리오에서 추적 정확도가 급격히 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 신경과학 및 인지과학의 통찰을 빌려, 객체 자체의 특징뿐만 아니라 객체 주변의 맥락 정보(Contextual Information)를 통합함으로써 시각적 모호성을 해소하고 객체 매칭의 정확도를 높이는 CAVIS 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체를 식별할 때 객체 내부의 특징만 보는 것이 아니라, 해당 객체가 처한 주변 환경(Context)을 함께 고려하는 것이다. 예를 들어, 단순히 '자전거'라는 객체만 식별하는 대신 '사람이 자전거를 타고 있는 상황'이라는 맥락을 함께 파악함으로써, 가려짐이 발생하더라도 더 정확하게 동일 객체를 추적할 수 있다는 직관에 기반한다.

이를 구현하기 위해 다음과 같은 두 가지 핵심 요소를 제안한다.

1. **Context-Aware Instance Tracker (CAIT):** 객체의 경계 주변 정보를 추출하여 핵심 인스턴스 특징과 결합하고, 이를 Transformer 기반의 추적 구조에 통합하여 맥락 인식형 매칭을 수행한다.
2. **Prototypical Cross-frame Contrastive (PCC) Loss:** 픽셀 임베딩을 기반으로 인스턴스별 프로토타입을 생성하고, 프레임 간의 특징 일관성을 강제함으로써 매칭 정확도를 향상시킨다.

## 📎 Related Works

기존의 VIS 방법론은 크게 MaskTrack R-CNN과 같은 초기 heuristic 기반 방식에서 시작하여, 최근에는 Mask2Former와 같은 Query-based 네트워크를 활용한 방식으로 발전하였다. 특히 온라인(online) 방식과 오프라인(offline) 방식으로 나뉘며, 최근에는 contrastive learning을 통해 프레임 간 특징 일관성을 높이려는 시도가 많았다.

추가적인 힌트를 사용하여 추적 성능을 높이려는 연구(예: 3D pose, optical flow 활용)들이 있었으며, 특히 CAROQ와 같은 연구는 픽셀 디코더에서 추출한 다단계 이미지 특징을 메모리 뱅크 형태로 사용하는 context feature 방식을 제안하였다. 그러나 이러한 전체 맥락 기반 접근법은 계산 복잡도가 높고 메모리 사용량이 많다는 단점이 있다.

반면, 본 논문의 CAVIS는 전체 맵이 아닌 각 객체 주변의 국소적인 맥락 정보에 집중함으로써 메모리 효율성을 확보하는 동시에 추적 성능을 높였다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

CAVIS의 전체 구조는 Mask2Former를 기반으로 한 segmentation 네트워크($S$)와 이를 통해 추출된 특징을 연결하는 tracking 네트워크($T$)로 구성된다. 전체 과정은 '맥락 인식 특징 추출 $\rightarrow$ 맥락 인식 매칭 $\rightarrow$ 일관성 손실 함수를 통한 학습' 순으로 진행된다.

### 2. Context-Aware Instance Tracker (CAIT)

#### (1) 맥락 인식 특징 추출 (Context-aware feature extraction)

먼저 Mask2Former를 통해 특징 맵 $F$, 인스턴스 특징 $\hat{Q}$, 마스크 $M$을 생성한다. 이후 객체의 경계 주변 정보를 캡처하기 위해 다음과 같은 절차를 거친다.

- 마스크 $M$에 Laplacian 필터를 적용하여 경계 영역을 찾고, 특징 맵 $F$에 $9 \times 9$ Average 필터를 적용하여 $\bar{F}$를 얻는다.
- 이를 통해 객체 주변의 특징인 $\tilde{Q}$를 다음과 같이 계산한다.
$$\tilde{Q}_{n}^{t} = \frac{\sum_{h=1}^{H} \sum_{w=1}^{W} \bar{F}_{\{h,w\}}^{t} * \mathbb{1}(\acute{M}_{\{n,h,w\}}^{t} > 0)}{\sum_{h=1}^{H} \sum_{w=1}^{W} \mathbb{1}(\acute{M}_{\{n,h,w\}}^{t} > 0)}$$
- 최종적으로 핵심 특징 $\hat{Q}$와 주변 특징 $\tilde{Q}$를 결합(Concatenate)한 후 MLP를 통과시켜 맥락 인식 인스턴스 특징 $Q$를 생성한다.
$$Q_{n}^{t} = \text{MLP}(\text{Concat}(\hat{Q}_{n}^{t}, \tilde{Q}_{n}^{t}))$$

#### (2) 맥락 인식 인스턴스 매칭 (Context-aware instance matching)

Transformer 기반의 추적 네트워크에서 기존의 Cross-Attention을 수정하여, Query와 Key에는 맥락 인식 특징($Q$)을 사용하고, Value에는 원래의 인스턴스 특징($\hat{Q}$)을 사용한다.
$$\text{Attn}(Q_{t-1}^{*}, Q_{t}, \hat{Q}_{t}) = \text{softmax} \left( \frac{Q_{t-1}^{*} \cdot (Q_{t})^{T}}{\sqrt{C}} \right) \hat{Q}_{t}$$
여기서 $Q^*$는 정렬된 특징을 의미하며, $\tilde{Q}$의 정렬을 위해 Hungarian matching 알고리즘을 사용한다.

### 3. Prototypical Cross-frame Contrastive (PCC) Loss

픽셀 레벨의 일관성을 유지하기 위해, 예측된 마스크를 사용하여 인스턴스별 프로토타입 $\eta$를 생성한다. 프로토타입은 해당 객체 영역에 속하는 픽셀 임베딩들의 평균값으로 정의된다.
$$\eta_{n}^{t} = \frac{\sum_{h=1}^{H} \sum_{w=1}^{W} F_{\{h,w\}}^{t} * \mathbb{1}(M_{\{h,w\}}^{t} == 1)}{\sum_{h=1}^{H} \sum_{w=1}^{W} \mathbb{1}(M_{\{h,w\}}^{t} == 1)}$$
이 프로토타입 $\eta$에 대해 프레임 간 Contrastive Loss를 적용함으로써, 동일 객체의 픽셀 임베딩이 프레임이 바뀌어도 일관된 표현력을 갖도록 강제한다.

### 4. 학습 절차 및 손실 함수

전체 segmentation 네트워크 $S$의 손실 함수는 다음과 같이 정의된다.
$$L_{S} = L_{VIS} + \lambda_{CTX} L_{CTX} + \lambda_{PCC} L_{PCC}$$
여기서 $L_{VIS}$는 표준 VIS 손실(분류, 마스크 BCE, Dice loss)이며, $L_{CTX}$는 맥락 인식 특징에 대한 contrastive loss, $L_{PCC}$는 위에서 설명한 프로토타입 기반 일관성 손실이다. 추적 네트워크 $T$는 처음 등장한 객체들에 대해 매칭 비용을 최소화하는 방향으로 학습된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** YouTube-VIS 2019, 2021, OVIS, VIPSeg.
- **백본 네트워크:** ResNet-50, Swin-L, ViT-L.
- **평가 지표:** Average Precision (AP), Average Recall (AR), Video Panoptic Quality (VPQ), Segmentation and Tracking Quality (STQ).

### 2. 주요 결과

- **VIS 성능:** 모든 벤치마크 데이터셋에서 SOTA(State-of-the-art)를 달성하였다. 특히 가려짐이 심한 OVIS 데이터셋에서 DVIS++ 대비 AP를 3.7 포인트 향상시키며 압도적인 성능을 보였다.
- **VPS 성능:** VIPSeg 데이터셋에서도 최고의 성능을 기록하였다. 특히 'thing' 클래스에 대한 성능($VPQ_{Th}$)이 크게 향상되었는데, 이는 맥락 인식 매칭 전략이 동적인 객체 추적에 매우 효과적임을 입증한다.
- **정성적 결과:** Figure 1과 3에서 보이듯, 기존 모델들이 Occlusion 이후 객체를 놓치거나 다른 객체로 오인하는 반면, CAVIS는 주변 맥락을 활용해 정확하게 동일 객체를 계속 추적하는 모습을 보인다.

### 3. Ablation Study

- **구성 요소의 효과:** $L_{CTX}$와 $L_{PCC}$를 모두 적용했을 때 가장 높은 성능을 보였으며, 특히 맥락 인식 매칭을 적용했을 때 AP가 2.3 포인트 추가 상승하였다.
- **필터 크기:** $9 \times 9$ 크기의 Average 필터가 최적의 성능을 보였다. 너무 큰 필터는 주변 정보를 지나치게 일반화하여 변별력을 떨어뜨리는 것으로 분석된다.
- **학습 프레임 수:** 인접한 3개의 프레임을 사용할 때 성능이 가장 좋았으며, 프레임 간격이 넓어질수록 주변 정보의 변화가 심해져 매칭이 어려워짐을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

CAVIS는 객체 중심의 특징 추출에서 벗어나 '주변 환경'이라는 추가 정보를 효율적으로 통합함으로써 VIS의 고질적인 문제인 Occlusion과 유사 객체 문제를 해결하였다. 특히 t-SNE 시각화 결과, 기존 모델보다 객체별 임베딩 클러스터가 훨씬 더 뚜렷하게 구분되는 것을 통해 맥락 정보가 객체의 변별력을 크게 높였음을 알 수 있다.

### 한계 및 논의

본 모델의 성능은 전제적으로 pretrained segmentation 네트워크의 마스크 예측 정확도에 의존한다. 만약 초기 분할 결과가 매우 부정확하다면, 이를 기반으로 추출되는 주변 맥락 정보 역시 오염될 가능성이 있다. 다만, 저자들은 Figure 4를 통해 마스크가 다소 부정확하더라도 맥락 모델링이 어느 정도 견고하게 추적을 수행할 수 있음을 보여주었다.

또한, 추론 속도 면에서 DVIS나 GenVIS 대비 약 5~7ms 정도의 추가 시간이 소요된다. 하지만 얻어지는 AP 상승폭에 비해 계산 비용 증가가 매우 적으므로, 이는 합리적인 trade-off라고 판단된다.

## 📌 TL;DR

본 논문은 객체 주변의 맥락 정보를 활용해 추적 성능을 높이는 **CAVIS** 프레임워크를 제안한다. **CAIT(Context-Aware Instance Tracker)**를 통해 주변 특징을 추출하고, **PCC Loss**를 통해 픽셀 레벨의 프레임 간 일관성을 확보하였다. 실험 결과, 특히 가려짐이 심한 복잡한 비디오 환경(OVIS 데이터셋 등)에서 기존 SOTA 모델들을 크게 상회하는 성능을 보였으며, 이는 VIS 및 VPS 작업 전반에 걸쳐 맥락 정보의 중요성을 입증한 연구이다.
