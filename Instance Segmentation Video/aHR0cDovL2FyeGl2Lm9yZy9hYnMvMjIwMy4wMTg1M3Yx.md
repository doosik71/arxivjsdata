# Efficient Video Instance Segmentation via Tracklet Query and Proposal

Jialian Wu, Sudhir Yarram, Hui Liang, Tian Lan, Junsong Yuan, Jayan Eledath, Gérard Medioni (2022)

## 🧩 Problem to Solve

본 논문은 비디오 내의 여러 객체 인스턴스를 동시에 분류, 세그멘테이션 및 추적하는 Video Instance Segmentation (VIS) 문제를 다룬다. 기존의 VIS 접근 방식은 크게 두 가지로 나뉘는데, 각각 다음과 같은 한계점이 존재한다.

1. **Frame-level 방식 (Tracking-by-segmentation):** 각 프레임에서 인스턴스 세그멘테이션을 수행한 후 데이터 연관(Data Association, DA) 알고리즘을 통해 추적한다. 이는 DA 알고리즘의 복잡성이 높고, 시간적 문맥(Temporal Context) 활용이 제한적이어서 객체 가려짐(Occlusion) 현상에 취약하다.
2. **Clip-level 방식:** 짧은 비디오 클립 단위로 처리하여 더 넓은 시간적 수용 영역을 활용한다. 하지만 대부분의 방법론이 end-to-end 학습이 불가능하거나 추론 속도가 느려 실시간 처리가 어렵다. 특히 최근 제안된 VisTR과 같은 Transformer 기반 방식은 프레임 단위의 Dense Attention으로 인해 학습 수렴 속도가 매우 느리며, 긴 비디오를 처리할 때 클립 간 인스턴스를 연결하기 위해 여전히 수동으로 설계된(Hand-crafted) 데이터 연관 과정이 필요하다는 단점이 있다.

따라서 본 논문의 목표는 학습과 추론 모두 효율적이면서, 데이터 연관 과정 없이 전체 비디오를 하나의 end-to-end 패스로 처리할 수 있는 실시간 VIS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Tracklet Query**와 **Tracklet Proposal**을 도입하여 공간과 시간 영역에서 객체 인스턴스를 효율적으로 표현하고 연관시키는 것이다.

- **RoI-wise 디자인:** 전체 픽셀에 대해 Dense Attention을 수행하는 대신, Tracklet Proposal로 정의된 RoI(Region of Interest) 영역 내에서만 쿼리와 비디오 특징을 상호작용하게 하여 연산 중복을 줄이고 학습 수렴 속도를 획기적으로 높였다.
- **Factorised Temporo-Spatial Self-Attention (FTSA):** 시간축과 공간축에 대해 분리된 Self-Attention을 적용하여, 하나의 쿼리가 시간적/공간적으로 동일한 인스턴스를 일관되게 추적하고 다른 인스턴스와의 관계를 추론할 수 있게 한다.
- **Temporal Dynamic Convolution (TDC):** 쿼리 임베딩을 기반으로 동적 컨볼루션 필터를 생성하고, 이를 3D 형태로 적용하여 주변 프레임의 시간적 문맥 정보를 효과적으로 수집한다.
- **Correspondence Learning:** 클립 간의 인스턴스 연결을 위해 수동 DA 대신, 이전 클립의 출력 쿼리를 다음 클립의 입력 쿼리로 사용하는 학습 방식을 제안하여 전체 비디오에 대해 완전히 end-to-end 학습 가능한 구조를 구현하였다.

## 📎 Related Works

기존의 VIS 연구는 크게 프레임 레벨과 클립 레벨로 구분된다.

- **Frame-level VIS:** MaskTrack R-CNN, QueryInst 등이 대표적이며, 이미지 세그멘테이션 후 추적하는 방식을 취한다. 하지만 이러한 방식은 프레임 간의 연관성을 찾기 위한 명시적인 DA 알고리즘에 의존하며, 비디오의 풍부한 시간적 정보를 충분히 활용하지 못한다는 한계가 있다.
- **Clip-level VIS:** MaskProp, VisTR 등이 있으며, 클립 내에서 정보를 전파하여 가려짐이나 모션 블러에 강건하다. VisTR은 DETR 구조를 확장하여 end-to-end 가능성을 제시했으나, 본 논문에서 지적하듯 Dense Attention으로 인한 느린 수렴 속도와 클립 간 연결을 위한 수동 DA의 필요성이 여전한 문제로 남아있다.

EfficientVIS는 이러한 기존 방식과 달리 RoI 기반의 상호작용을 통해 효율성을 확보하고, Correspondence Learning을 통해 클립 간 연결까지 신경망 내에서 학습함으로써 완전한 end-to-end 파이프라인을 달성하여 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
EfficientVIS는 CNN 백본을 통해 비디오 기본 특징(Base Feature)을 추출한 후, $M$번의 반복적인 쿼리-비디오 상호작용(Query-Video Interaction) 과정을 거친다. 각 반복 단계는 다음과 같이 구성된다.

1. **Query Interaction:** Factorised Temporo-Spatial Self-Attention (FTSA)를 통해 쿼리들 간의 정보를 교환한다.
2. **Video Interaction:** Temporal Dynamic Convolution (TDC)을 통해 비디오 특징에서 타겟 인스턴스 정보를 수집한다.
3. **Head Networks:** 업데이트된 쿼리와 특징을 바탕으로 마스크, 클래스, 프레임별 Proposal을 예측하고 쿼리를 갱신한다.

### 주요 구성 요소 및 방정식

#### 1. Tracklet Query & Proposal
- **Tracklet Query ($q_i$):** 인스턴스의 고유한 외관 정보를 인코딩하는 임베딩 벡터이다. 각 인스턴스 쿼리는 시간축 $T$에 대해 $T$개의 임베딩을 가진다 ($q_i \in \mathbb{R}^{T \times C}$). 이는 시간 경과에 따른 외관 변화를 반영하기 위함이다.
- **Tracklet Proposal ($b_i$):** 비디오 클립 전체에서 인스턴스의 위치를 나타내는 시공간 튜브(Space-time tube)로, $T$개 프레임의 바운딩 박스 집합이다 ($b_i \in \mathbb{R}^{T \times 4}$).

#### 2. Factorised Temporo-Spatial Self-Attention (FTSA)
쿼리 간의 대응 관계를 학습하기 위해 시간적 MSA와 공간적 MSA를 순차적으로 수행한다.
- **Temporal Self-Attention:** 동일한 인스턴스 쿼리 내에서 시간축으로 임베딩 간 정보를 교환하여, 동일 인스턴스를 시간적으로 일관되게 추적한다.
  $$\{q_t^i\}_{t=1}^T \leftarrow \text{MSA}(\{q_t^i\}_{t=1}^T), \quad i=1, \dots, N$$
- **Spatial Self-Attention:** 동일한 프레임 내에서 서로 다른 인스턴스 쿼리 간 정보를 교환하여 객체 간의 관계를 추론한다.
  $$\{q_t^i\}_{i=1}^N \leftarrow \text{MSA}(\{q_t^i\}_{i=1}^N), \quad t=1, \dots, T$$

#### 3. Temporal Dynamic Convolution (TDC)
쿼리 $q_t^i$로부터 동적 필터 $w_t^i$를 생성하고, 이를 RoIAlign으로 추출된 특징 맵에 적용하여 시간적 문맥을 수집한다.
$$o_t^i = \sum_{t' = t-1}^{t+1} a_{i,(t,t')} \circ \text{conv2d}(w_t^i, \phi(f_{t'}, b_{t'}^i))$$
여기서 $\phi$는 RoIAlign 연산이며, $a_{i,(t,t')}$는 코사인 유사도와 Softmax를 통해 계산된 적응형 가중치이다. 이를 통해 현재 프레임뿐만 아니라 인접 프레임의 특징을 함께 수집하여 가려짐 현상에 대응한다.

#### 4. 훈련 목표 및 손실 함수
Bipartite Matching(Hungarian Algorithm)을 통해 예측값 $y$와 정답 $\hat{y}$ 사이의 최적 할당 $\hat{\sigma}$를 찾은 후, 다음의 통합 손실 함수를 사용하여 학습한다.
$$L_{\text{clip}}(\hat{\sigma}) = \sum_{j=1}^G \sum_{t=1}^T [ -p_{\hat{\sigma}(j)}(\hat{c}_t^j) + L_{\text{CE}}(m_{t}^{\hat{\sigma}(j)}, \hat{m}_t^j) + \mathbb{1}_{\{\hat{c}_t^j \neq \emptyset\}} (L_{\text{box}}(b_{t}^{\hat{\sigma}(j)}, \hat{b}_t^j) + L_{\text{dice}}(m_{t}^{\hat{\sigma}(j)}, \hat{m}_t^j)) ]$$
- $L_{\text{CE}}$: Binary Cross-Entropy loss (마스크 및 클래스 분류)
- $L_{\text{dice}}$: Dice loss (마스크 정교화)
- $L_{\text{box}}$: IoU 및 $L_1$ distance (박스 회귀)

#### 5. Correspondence Learning
클립 간의 연결을 위해, 훈련 시 두 개의 클립을 쌍으로 묶어 처리한다. 첫 번째 클립의 출력 쿼리를 시간축으로 평균내어 두 번째 클립의 초기 쿼리로 입력한다. 이때 두 번째 클립은 첫 번째 클립에서 결정된 할당 $\hat{\sigma}$를 그대로 사용하여 학습함으로써, 쿼리가 클립이 바뀌어도 동일한 인스턴스를 추적하도록 강제한다.

## 📊 Results

### 실험 설정
- **데이터셋:** YouTube-VIS 2019 및 2021 벤치마크.
- **지표:** Video-level Average Precision (AP) 및 Average Recall (AR).
- **백본 및 설정:** ResNet-50, 클립 길이 $T=36$, 쿼리 개수 $N=10$, 반복 횟수 $M=6$.

### 주요 결과
- **정량적 성능:** ResNet-50 기반 EfficientVIS는 YouTube-VIS 2019에서 37.9 AP (multi-scale training 적용 시) 및 36 FPS의 속도를 기록하였다.
- **수렴 속도:** VisTR과 비교했을 때, 약 15배 적은 에포크(33 epochs)만으로도 더 높은 정확도에 도달하였다. 이는 RoI-wise 설계가 불필요한 배경 연산을 줄였기 때문이다.
- **강건성:** 프레임 속도를 1.5 FPS까지 낮춘 저프레임 비디오에서도 성능 하락이 거의 없음을 확인하였다. 이는 모션 기반 추적이 아닌 외관 쿼리 기반의 검색 방식을 사용하기 때문이다.
- **End-to-End 효율성:** 수동 DA를 사용하는 "Hand-craft" 방식이나 VisTR보다 본 논문의 fully end-to-end 방식이 더 높은 성능을 보였으며, 이는 학습 과정에서 최적의 연결 방식이 직접 최적화되었기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점
- **효율적인 학습과 추론:** RoI-wise 접근법을 통해 Transformer 계열 모델의 고질적인 문제인 느린 수렴 속도를 해결하고 실시간성을 확보하였다.
- **완전한 End-to-End 구조:** Correspondence Learning을 통해 클립 간의 복잡한 수동 매칭 과정을 제거함으로써 시스템의 단순성과 성능을 동시에 향상시켰다.
- **시간적 문맥 활용:** TDC와 FTSA를 통해 객체가 일시적으로 가려지더라도 주변 프레임의 정보를 이용하여 안정적으로 추적할 수 있음을 보였다.

### 한계 및 논의
- **메모리 제약:** 비디오 길이가 매우 길 경우 여전히 클립 단위로 나누어 처리해야 하며, 이때 쿼리를 전달하는 방식에 의존한다.
- **가정:** 본 연구는 쿼리가 인스턴스의 충분한 외관 정보를 담고 있다는 가정하에 작동하며, 완전히 새로운 객체가 갑자기 등장하는 상황에서의 쿼리 재할당 전략(Re-initialization)은 간단한 휴리스틱으로 처리되어 추가적인 연구가 필요할 수 있다.

### 비판적 해석
본 논문은 VisTR의 Dense Attention이 VIS 작업에 있어 과도한 중복 연산을 유발한다는 점을 정확히 짚어냈으며, 이를 RoI-wise 상호작용으로 해결한 점이 매우 인상적이다. 특히, 단순한 성능 향상을 넘어 학습 효율성(15배 빠른 수렴)을 입증함으로써 실제 연구 및 적용 단계에서의 개발 사이클을 크게 단축시켰다는 점에서 실용적 가치가 매우 높다고 평가된다.

## 📌 TL;DR

EfficientVIS는 Tracklet Query와 Proposal을 이용해 객체의 외관과 위치를 동시에 모델링하고, RoI-wise 상호작용과 Correspondence Learning을 통해 데이터 연관 과정 없이 전체 비디오를 처리하는 **최초의 완전한 end-to-end 실시간 VIS 프레임워크**이다. VisTR 대비 학습 속도를 15배 높이면서도 SOTA 수준의 정확도를 달성하였으며, 특히 저프레임 비디오 및 가려짐 상황에서 강건한 추적 성능을 보인다. 이 연구는 향후 실시간 비디오 분석 시스템 및 자율 주행 등 연속적인 객체 추적이 필요한 분야에 핵심적인 역할을 할 가능성이 크다.