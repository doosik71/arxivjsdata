# Object-Centric Multiple Object Tracking

Zixu Zhao et al. (2023)

## 🧩 Problem to Solve

본 논문은 다중 객체 추적(Multiple Object Tracking, MOT) 분야에서 발생하는 막대한 어노테이션 비용 문제와 기존 비지도 객체 중심 학습(Unsupervised Object-Centric Learning, OCL) 모델의 한계를 해결하고자 한다. 

전통적인 MOT 파이프라인은 객체 검출(Detection)을 위한 대량의 바운딩 박스 레이블과, 시간축에 따른 객체 연관성(Association)을 학습시키기 위한 ID 레이블을 필요로 하는 Label-intensive한 구조이다. 한편, 비지도 OCL 방법론은 추가적인 위치 정보 없이 장면을 엔티티(Entity)로 분할할 수 있어 MOT의 대안으로 주목받았으나, 다음과 같은 두 가지 핵심적인 문제가 존재한다.

1. **Part-Whole 문제**: 하나의 객체가 여러 개의 슬롯(Slot)으로 쪼개져 인식되는 현상이 발생한다.
2. **시간적 일관성(Temporal Consistency) 부족**: 동일한 객체가 시간에 따라 서로 다른 슬롯에 할당되어 ID Switch가 빈번하게 발생하는 문제가 있다.

따라서 본 연구의 목표는 ID 레이블 없이, 그리고 매우 적은 양의 검출 레이블만으로도 객체를 하나의 전체(Whole)로 인식하고 시간에 따라 일관되게 추적할 수 있는 비지도 객체 중심 MOT 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체 중심 표현을 관리하는 **Object Memory Module**과 슬롯을 정제하는 **Index-Merge Module**을 도입하여 OCL 모델의 고유한 한계를 극복하는 것이다.

- **Label-Efficient MOT**: ID 레이블이 전혀 필요 없으며, 객체 위치 및 특징 바인딩을 위해 매우 적은 양(0%~6.25%)의 검출 레이블만 사용한다.
- **Object Memory Module**: 과거의 객체 상태를 미래로 투영하는 Memory Rollout 메커니즘을 통해 폐색(Occlusion) 상황에서도 객체의 상태를 예측하고 시간적 일관성을 유지한다.
- **Index-Merge Module**: 슬롯과 메모리 버퍼 간의 이분법적 인터페이스를 구축하여, 여러 슬롯으로 분산된 객체의 부분들을 하나의 메모리 버퍼로 통합함으로써 Part-Whole 및 중복 슬롯 문제를 해결한다.
- **EM-inspired Loss**: 정답 ID 레이블 없이도 슬롯과 메모리 간의 최적 할당을 학습하기 위해 Expectation-Maximization(EM) 패러다임에서 영감을 받은 자기지도 학습 손실 함수를 제안한다.

## 📎 Related Works

### Unsupervised Object-centric Learning
Slot Attention 등을 통해 입력 데이터를 집합 구조의 병목(set-structured bottleneck)으로 인코딩하여 객체를 발견하는 연구들이다. 최근 SAVi, STEVE 등이 비디오 영역으로 확장되었으나, 이들은 주로 클러스터링 유사도 지표(FG-ARI 등)로 평가될 뿐, 실제 MOT에서 중요한 시간적 일관성이나 Part-Whole 문제에 대한 엄격한 제약이 부족했다.

### Self-supervised MOT
ID 레이블을 줄이기 위해 Cycle-consistent loss 등을 활용한 연구들이 존재한다. SORT나 IOU 같은 방법론은 휴리스틱한 큐(Kalman filter, IoU)를 사용하지만, 빈번한 폐색이나 카메라 움직임이 있는 상황에서 성능이 급격히 저하된다. 본 연구는 이러한 휴리스틱 방식 대신 학습 가능한 메모리 기반의 연관성 모델을 제안하여 차별점을 둔다.

### Memory Models
비디오 분석에서 외부 메모리를 통해 장기 정보를 저장하는 방식이다. MemTrack이나 SimTrack 등이 있으나 이는 주로 단일 객체 추적(SOT)에 집중되어 있다. MeMOT는 MOT를 위해 시공간 메모리를 구축했지만, 학습을 위해 비용이 많이 드는 ID 레이블을 필요로 한다. 반면, OC-MOT는 자기지도 방식의 Memory Rollout을 통해 ID 레이블 없이도 추적이 가능하게 설계되었다.

## 🛠️ Methodology

### 전체 파이프라인
OC-MOT는 크게 세 단계로 구성된다: **Object-centric Grouping $\rightarrow$ Memory Rollout $\rightarrow$ Index-Merge**.

### 1. Object-centric Grouping
Slot Attention을 사용하여 비디오 프레임의 인코더 특징을 슬롯 벡터 세트 $\{S^t\}$로 변환한다. 기본적으로 다음의 자기지도 재구성 손실 함수를 통해 학습된다.
$$L_{oc}^{rec} = ||y - \text{Dec}(S)||^2$$
여기서 $y$는 원본 픽셀 또는 특징 표현이며, 디코더는 객체를 개별 슬롯에 바인딩하도록 설계되었다.

### 2. Memory Module
추적 중인 모든 객체의 역사적 표현을 메모리 버퍼 $M \in \mathbb{R}^{M \times T \times d}$에 저장한다.
- **Memory Rollout**: 시간 $t$에서 과거 상태 $M_{<t}$를 기반으로 현재 시점의 객체 표현 $\tilde{M}_t$를 예측한다.
$$\tilde{M}_t = \text{Rollout}(M_{<t})$$
이 과정에는 약 1.6M 파라미터의 소형 GPT-2 모델이 사용되며, 자기회귀 트랜스포머(auto-regressive transformer)를 통해 시간적 추론을 수행한다.

### 3. Index-Merge Module
슬롯 $S^t$와 메모리 $\tilde{M}_t$ 사이의 이산적 인터페이스 역할을 수행한다.
- **Slot-to-memory Index**: 슬롯을 쿼리($q$)로, 롤아웃 결과 $\tilde{M}_t$를 키와 값($k, v$)으로 하는 Multi-Head Attention(MHA)을 통해 소프트 할당 행렬 $I_t \in \mathbb{R}^{N \times M}$를 생성한다.
$$I_t = \text{MHA}(k, v = \tilde{M}_t, q = S^t).\text{attnweight}$$
- **Memory-to-slot Merge**: $I_t$를 마스크로 사용하여, 동일한 버퍼에 할당된 슬롯들을 통합해 최종 객체 표현 $m_t$를 생성한다.
$$m_t = \text{MHA}(k, v = S^t, q = \tilde{M}_t, \text{attnmask} = I_t)$$

### 4. Model Training under EM Paradigm
ID 레이블이 없으므로, 슬롯 $S_i^t$와 메모리 $M_j^t$ 사이의 할당 비용 $L_{assign}$을 최소화하는 방향으로 학습한다.
$$L_{assign}(S_i^t, M_j^t) = \lambda_1 \text{BCELoss}(\text{Dec}(S_i^t), \text{Dec}(M_j^t)) + \lambda_2 ||\text{Dec}(S_i^t) - \text{Dec}(M_j^t)||^2 + \lambda_3 ||S_i^t - M_j^t||^2$$
비미분 가능한 $\text{argmax}$ 대신 EM 패러다임을 적용하여, 모든 가능한 할당의 기댓값을 최적화하는 손실 함수를 정의한다.
$$\mathcal{L} = \sum_{t=1}^T \sum_{i=1}^N \sum_{j=1}^M I_t[i, j] L_{assign}(S_i^t, M_j^t)$$

### 5. Model Inference
추론 시에는 $I_t$를 이진화하여 $\text{argmax}$ 기반의 하드 할당을 수행한다.
- **Object-in**: 첫 프레임에서는 IoU 임계값 $\tau_{iou}$를 기준으로 중복 슬롯을 제거하여 버퍼를 초기화한다. 이후 프레임에서는 롤아웃 결과와 IoU가 낮은 새로운 슬롯이 발견되면 새로운 버퍼를 활성화한다.
- **Object-out**: 객체가 소실되거나 폐색된 경우 $\tau_{out}$ 프레임 동안 버퍼를 유지하며, 이를 초과하면 버퍼를 종료한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CATER(합성 비디오), FISHBOWL(수족관 영상), KITTI(실제 주행 영상).
- **지표**: IDF1(추적 일관성), MOTA(객체 커버리지), MT, ML, IDS(ID 전환 횟수), Track mAP.
- **비교 대상**: SAVi(객체 중심 모델), IOU, SORT, Visual-Spatial(비지도 MOT), MOTR(완전 지도 MOT).

### 주요 결과
1. **CATER 데이터셋**: OC-MOT는 비지도 베이스라인들을 압도하며, 완전히 지도 학습된 MOTR에 근접하는 성능을 보였다. 특히 ID Switch(IDS)를 획기적으로 줄여 시간적 일관성을 확보했음을 증명했다.
2. **FISHBOWL 데이터셋**: 복잡한 배경과 심한 폐색이 존재하는 환경에서도 OC-MOT는 최첨단 비지도 추적 성능을 기록했다. 특히 폐색 상황에서 IoU 기반(SORT, IOU) 방식보다 훨씬 강력한 연관성 능력을 보여주었다.
3. **분석**: SAVi와 같은 기존 OCL 모델은 객체를 과분할(Over-segmentation)하여 False Positive가 높았으나, OC-MOT는 Index-Merge 모듈을 통해 이를 해결하였다.

| Method | Detection Label | ID Label | IDF1 $\uparrow$ (CATER) | MOTA $\uparrow$ (CATER) | IDS $\downarrow$ (CATER) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SAVi | 0% | No | 73.2% | 52.5% | 20352 |
| SORT | 84.5% | No | 79.2% | 71.8% | 219 |
| **OC-MOT** | **0%** | **No** | **88.6%** | **82.4%** | **5658** |
| MOTR | 100% | Yes | 89.3% | 83.3% | 66 |

*(참고: OC-MOT는 매우 적은 레이블로도 지도 학습 모델인 MOTR의 성능에 상당히 근접함)*

## 🧠 Insights & Discussion

### 강점
- **레이블 효율성**: ID 레이블 없이도 고성능 추적이 가능하며, 검출 레이블을 6.25%만 사용하고도 상당한 성능 향상을 이루었다.
- **폐색 대응**: Memory Rollout을 통해 객체가 일시적으로 가려지더라도 그 상태를 예측하여 추적을 유지하는 능력이 탁월하다.
- **구조적 정제**: Index-Merge 설계를 통해 OCL 모델의 고질적인 문제인 Part-Whole 문제를 효과적으로 완화했다.

### 한계 및 비판적 해석
- **해상도 문제**: KITTI 실험에서 나타났듯, DINOSAUR와 같은 OCL 백본이 특징 맵을 16배 다운샘플링하여 사용하기 때문에 멀리 있는 작은 객체의 마스크 정밀도가 떨어진다. 이는 MOTA 수치에 부정적인 영향을 미친다.
- **End-to-End 부재**: 현재 모델은 사전 학습된 OCL 모델을 플러그 앤 플레이 형태로 사용한다. 그룹핑 모듈과 메모리 모듈을 통합하여 엔드-투-엔드로 학습시킨다면 성능이 더 향상될 가능성이 있다.
- **과분할 의존성**: 배경을 객체로 인식하는 과분할 문제를 완전히 해결하기 위해 여전히 소량의 시맨틱 레이블(DETR loss 등)에 의존하고 있다.

## 📌 TL;DR

본 논문은 비지도 객체 중심 학습(OCL)의 고질적인 문제인 **객체 분할(Part-whole)**과 **시간적 불일치(ID Switch)**를 해결하기 위해 **Object Memory**와 **Index-Merge** 모듈을 도입한 OC-MOT를 제안한다. ID 레이블 없이 EM 기반의 자기지도 학습을 통해 객체 연관성을 학습하며, 매우 적은 양의 검출 레이블만으로도 기존 비지도 추적기보다 훨씬 뛰어난 성능을 보였으며 지도 학습 모델(MOTR)에 근접하는 결과를 달성했다. 이 연구는 MOT 분야에서 고비용의 ID 어노테이션을 대체할 수 있는 효율적인 프레임워크를 제시했다는 점에서 큰 의의가 있다.