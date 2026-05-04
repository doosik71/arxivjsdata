# Human Instance Segmentation and Tracking via Data Association and Single-stage Detector

Lu Cheng, Mingbo Zhao (2022)

## 🧩 Problem to Solve

본 논문은 비디오 내에서 사람의 인스턴스를 픽셀 수준으로 분할하고 동일한 인물을 추적하는 Human Video Instance Segmentation (HVIS) 문제를 해결하고자 한다. HVIS는 가상 현실의 인간 모델링, 비디오 감시 및 비디오 처리 등 다양한 분야에서 중요하게 활용된다.

기존의 Video Instance Segmentation (VIS) 방법론들은 주로 Mask-RCNN과 같은 2단계(Two-stage) 검출기 프레임워크에 기반하고 있다. 이러한 방식은 데이터 매칭을 위해 타겟의 외형(Appearance)과 모션 정보를 처리하는 과정에서 계산 비용이 크게 증가하며, 이는 실시간 성능 구현에 걸림돌이 된다. 또한, 기존의 VIS 데이터셋들은 비디오에 등장하는 모든 사람을 포괄적으로 다루지 않는다는 한계가 있다.

따라서 본 연구의 목표는 단일 단계(Single-stage) 검출기를 기반으로 하여 계산 효율성을 높이면서도, 데이터 연관(Data Association) 전략을 통해 정확한 추적 및 분할 성능을 확보하는 HVIS 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어와 기여 사항은 다음과 같다.

1.  **단일 단계 프레임워크의 통합**: SOLO (Segmenting Objects by Locations) 프레임워크에 특징 추출 및 데이터 연관 구성 요소를 통합하였다. 이를 통해 타겟 인식, 분할, 추적을 동시에 수행하며, 제안된 구조는 객체의 크기에 관계없이 동일한 해상도로 마스크를 예측하여 경계선 표현 능력을 높였다.
2.  **강건한 특징 추출 전략**: Siamese 네트워크 구조를 통해 인스턴스의 임베딩을 추출하며, 특히 **Centroid Sampling Strategy**를 도입하였다. 이는 객체가 심하게 겹쳐 있는 상황에서도 마스크 내부의 최대 면적 윤곽선 중심을 샘플링함으로써, 갑작스러운 활동 변화 시에도 임베딩 위치가 마스크 밖으로 벗어나 ID Switch가 발생하는 문제를 완화한다.
3.  **PVIS 데이터셋 구축**: 기존의 YouTube-VIS, DAVIS, DAVSOD 데이터셋을 재라벨링하고 정제하여 사람 전용 비디오 인스턴스 분할 데이터셋인 PVIS를 구축함으로써, 연구 분야의 데이터 부족 문제를 해결하였다.

## 📎 Related Works

논문에서는 VIS 접근 방식을 크게 두 가지로 분류하여 설명한다.

1.  **Mask Propagation 기반 방법**: 첫 번째 프레임의 마스크를 가이드로 삼아 이후 프레임으로 전파하는 방식이다. 장기적인 VIS 문제 해결에 유리하지만, 첫 프레임의 분할 정확도에 지나치게 의존하며, 빠르게 위치나 외형이 변하는 경우 성능이 급격히 저하된다.
2.  **Tracking-by-Detection 기반 방법**: 각 프레임에서 인스턴스를 독립적으로 분할한 후 데이터 연관 방법을 통해 매칭하는 방식이다. 주로 Mask-RCNN 기반의 2단계 검출기를 사용하는데, 이는 각 하위 작업(분할, 임베딩 추출 등)마다 별도의 모델을 설계하고 튜닝해야 하며, 계산 비용이 높아 실시간 성능 확보가 어렵다.

본 제안 방법은 SOLO라는 단일 단계 검출기를 채택함으로써 2단계 방식의 복잡성을 제거하고, 데이터 연관 과정을 효율적으로 통합하여 기존의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
본 시스템은 크게 **특징 추출 모듈(Feature Extraction Module)**, **유사도 추정 모듈(Similarity Estimation Module)**, 그리고 **매칭 모듈(Matching Module)**로 구성된다.

### 주요 구성 요소 및 상세 설명

#### 1. 특징 추출 모듈 (Feature Extraction Module)
인스턴스를 대표하는 외형 임베딩을 추출하기 위해 ResNet-101 백본 네트워크를 사용한다. 
- **멀티 스케일 특징 추출**: 스케일 변화에 강건한 특징을 얻기 위해 FPN 구조 대신 백본의 여러 계층에서 직접 11개의 특징 맵을 선택하여 추출한다.
- **Centroid Sampling Strategy**: 단순한 바운딩 박스 중심점 샘플링은 객체 간 겹침이 심할 때 모호한 양성 샘플(Fuzzy positive sample) 문제를 야기한다. 이를 해결하기 위해 최대 면적 윤곽선의 중심점 $(x_c, y_c)$를 계산하여 샘플링한다. 중심점 계산식은 다음과 같다.

$$m_{pq} = \sum_{y=1}^{H_m} \sum_{x=1}^{W_m} x^p \cdot y^q \cdot ct_{max}(x, y)$$
$$x_c = \frac{m_{10}}{m_{00}}, \quad y_c = \frac{m_{01}}{m_{00}}$$

여기서 $ct_{max}$는 해당 인스턴스의 최대 마스크 면적에 해당하는 이진 이미지이다.

#### 2. 유사도 추정 모듈 (Similarity Estimation Module)
두 프레임 간의 임베딩 유사도를 계산하여 동일 인스턴스일 확률을 구한다.
- **구조**: Siamese 네트워크를 통해 추출된 두 프레임의 특징 행렬 $E_t, E_{t-n} \in \mathbb{R}^{e \times N_m}$을 결합하여 텐서 $T_{t, t-n} \in \mathbb{R}^{2e \times N_m \times N_m}$를 생성하고, 이를 $1 \times 1$ 컨볼루션 커널을 통해 유사도 행렬로 압축한다.
- **동적 상황 처리**: 인스턴스의 진입과 퇴장을 처리하기 위해 행과 열에 벡터를 추가하여 $M_{fw} \in \mathbb{R}^{(N_m+1) \times N_m}$ (전방 연관) 및 $M_{rv} \in \mathbb{R}^{N_m \times (N_m+1)}$ (역방향 연관)를 구성하고 Softmax를 통해 확률 행렬 $P_{fw}, P_{rv}$를 얻는다.

#### 3. 손실 함수 및 학습 절차
학습은 분할, 검출, 매칭 세 가지 손실의 합으로 이루어지며, Multi-task learning 기법을 통해 가중치를 자동으로 조절한다.
- **매칭 손실 ($L_{match}$)**: 전방 손실($L_{fw}$), 역방향 손실($L_{rv}$), 그리고 비최대값 손실($L_{nm}$)의 평균으로 계산된다.
$$L_{match} = \frac{1}{3}(L_{fw} + L_{rv} + L_{nm})$$
전방/역방향 손실은 다음과 같이 정의된다.
$$L^* = \frac{\sum_{i=1}^{row} \sum_{j=1}^{col} (G^*(i, j) (-\log P^*(i, j)))}{\sum_{i=1}^{row} \sum_{j=1}^{col} G^*(i, j)}$$

#### 4. 추적 궤적 생성 (Tracking Trajectory Generation)
최종 매칭은 다음의 3단계 과정을 거쳐 수행된다.
1.  **Kalman Filter**: 모션 트렌드가 일치하지 않는 결과를 1차적으로 필터링한다.
2.  **특징 임베딩 연관**: 유사도 행렬의 중앙값(Median)을 사용하여 외형 정보를 기반으로 매칭한다.
3.  **Mask IOU 연관**: 특징 매칭에 실패한 경우, 마스크 간의 겹침 정도를 계산하여 최종 매칭한다.
$$\text{MaskIOU} = \frac{\text{Area}_{dt} \cap \text{Area}_{tr}}{\text{Area}_{dt} \cup \text{Area}_{tr}}$$
이후 헝가리안 알고리즘(Hungarian algorithm)을 통해 일대일 매칭을 완료한다.

## 📊 Results

### 실험 설정
- **데이터셋**: PVIS (1,117개 비디오, 2,025개 객체, 약 6만 개의 마스크).
- **평가 지표**: HOTA (Higher Order Tracking Accuracy)를 주 지표로 사용하며, DetA(검출 정확도), AssA(연관 정확도), IDS(ID 스위치 횟수) 등을 함께 측정한다.
- **비교 대상**: MaskTrack-RCNN (2-stage), SipMask (1-stage).

### 정량적 결과
실험 결과, 제안 방법이 모든 지표에서 가장 우수한 성능을 보였다.

| Methods | Type | HOTA | DetA | AssA | IDS |
| :--- | :--- | :---: | :---: | :---: | :---: |
| MaskTrack-RCNN | Two-stage | 54.3 | 42.0 | 72.8 | 91 |
| SipMask | One-stage | 51.9 | 38.7 | 72.2 | 76 |
| **Ours** | **One-stage** | **58.7** | **45.8** | **77.2** | **65** |

### 주요 분석 결과 (Ablation Study)
- **특징 계층 수**: 6개 계층보다 11개 계층을 사용할 때 AssA가 향상되어, 다양한 스케일의 정보를 캡처하는 것이 ID 매칭에 유리함이 증명되었다.
- **샘플링 전략**: 단순 Bbox 중심 샘플링보다 제안된 Centroid Sampling이 HOTA를 0.9% 상승시켰으며, 특히 객체 간 겹침 상황에서 ID Switch를 효과적으로 줄였다.
- **연관 방법**: Kalman Filter와 Mask IOU를 모두 사용했을 때 가장 높은 HOTA(58.7)를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 단일 단계 검출기인 SOLO를 확장하여 HVIS를 효율적으로 해결하였다. 특히 2단계 검출기가 가진 무거운 연산 비용 문제를 해결하면서도, 멀티 스케일 특징 추출과 정교한 중심점 샘플링 전략을 통해 추적의 강건성을 확보한 점이 인상적이다.

**강점**: 
- 단일 단계 구조를 통해 실시간성에 가까운 효율성을 확보하였다.
- Centroid sampling과 3단계 매칭 프로세스(Kalman $\rightarrow$ Feature $\rightarrow$ Mask IOU)를 통해 폐쇄(Occlusion) 및 겹침 상황에서의 ID 유지 능력을 크게 향상시켰다.

**한계 및 논의**:
- 논문에서 인스턴스의 최대 개수를 $N_m = 50$으로 고정하여 학습 및 추론을 진행한다. 이는 매우 밀집된 군중 장면에서 성능 저하를 일으킬 가능성이 있다.
- 특징 계층 수를 늘릴 때 검출 정확도(DetA)와 추적 정확도(AssA) 사이에 상충 관계(Trade-off)가 발생한다는 점이 언급되었는데, 이에 대한 근본적인 원인 분석과 최적의 균형점을 찾는 추가 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 **SOLO 단일 단계 검출기**를 기반으로 **Siamese 특징 추출**과 **Centroid Sampling 전략**을 통합한 새로운 Human Video Instance Segmentation 방법론을 제안하였다. 또한, 사람 전용 VIS 데이터셋인 **PVIS**를 구축하여 성능을 검증하였으며, 제안 방법은 기존 MaskTrack-RCNN 등의 2단계 방식보다 효율적이면서도 높은 HOTA 성능(58.7)을 달성하였다. 이 연구는 실시간성이 요구되는 인간 행동 분석 및 비디오 감시 시스템에 직접적으로 적용될 가능성이 높다.