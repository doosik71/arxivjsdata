# Robust Online Video Instance Segmentation with Track Queries

Zitong Zhan, Daniel McKee, Svetlana Lazebnik (2022)

## 🧩 Problem to Solve

본 논문은 비디오 인스턴스 분할(Video Instance Segmentation, VIS) 작업에서 기존의 고성능 모델들이 가진 실용적 한계를 해결하고자 한다. 최근 Transformer 기반의 방법론들이 우수한 성능을 보이고 있으나, 대부분의 최신 모델들은 비디오 클립 전체를 한 번에 처리하여 마스크 볼륨을 예측하는 오프라인(Offline) 방식으로 동작한다.

이러한 오프라인 방식은 다음과 같은 심각한 문제점을 야기한다. 첫째, 실시간 처리가 불가능하여 실제 환경에 적용하기 어렵다. 둘째, 비디오 전체 프레임에 대해 Transformer 추론을 수행할 때 발생하는 막대한 GPU 메모리 요구량으로 인해, 약 50프레임을 초과하는 긴 비디오를 처리할 수 없다. 결과적으로 UVO나 OVIS와 같이 175프레임 이상의 긴 비디오를 포함하거나, 심한 가려짐(Occlusion) 및 복잡한 군중 장면이 등장하는 최신 벤치마크 데이터셋에서 기존 오프라인 모델들은 제대로 작동하지 않는다. 따라서 본 논문의 목표는 오프라인 모델에 필적하는 정확도를 유지하면서도, 메모리 제약 없이 긴 비디오를 처리할 수 있는 효율적인 완전 온라인(Fully-online) Transformer 기반 VIS 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 강력한 정지 이미지 분할 모델인 Mask2Former와 다중 객체 추적(Multi-Object Tracking, MOT) 분야에서 제안된 Track Queries 메커니즘을 결합하는 것이다.

ROVIS(Robust Online Video Segmentation)는 Mask2Former의 구조를 기반으로 하되, 프레임 간에 객체 정보를 전달하기 위한 경량화된 메커니즘인 Track Queries를 도입하였다. 이는 고비용의 시공간 어텐션(Spatio-temporal attention)을 사용하는 대신, 이전 프레임에서 탐지된 객체의 쿼리 임베딩을 다음 프레임으로 직접 전달함으로써 객체의 정체성을 유지하도록 설계되었다. 이를 통해 별도의 복잡한 연결 임베딩(Linking embedding)이나 최적화 단계 없이도 효율적인 온라인 추적과 분할이 가능함을 입증하였다.

## 📎 Related Works

기존의 VIS 접근 방식은 크게 두 가지 흐름으로 나뉜다. 초기에는 각 프레임에서 객체를 탐지한 후 유사도 기반으로 연결하는 Tracking-by-detection 방식의 온라인 모델(예: MaskTrack R-CNN)이 주를 이루었다. 이후 VisTR, SeqFormer와 같은 오프라인 Transformer 기반 모델들이 등장하여 시공간 어텐션을 통해 3D 마스크 볼륨을 예측함으로써 YouTube-VIS 2019와 같은 짧은 비디오 데이터셋에서 압도적인 성능을 기록하였다.

최근에는 효율성을 위해 IDOL이나 MinVIS와 같은 온라인 Transformer 기반 방법들이 제안되었다. IDOL은 판별적인 인스턴스 임베딩을 학습하여 메모리 뱅크 기반으로 연결을 수행하며, MinVIS는 Mask2Former의 객체 쿼리 간 이분 매칭(Bipartite matching)을 통해 단순하게 연결을 수행한다.

본 연구는 특히 TrackFormer의 Track Query 개념에 주목한다. 기존 TrackFormer는 vanilla DETR 기반의 탐지기를 사용했기에 VIS의 다양하고 복잡한 외형 변화를 처리하기에는 탐지 성능이 부족했다. ROVIS는 이를 개선하여 훨씬 강력한 Mask2Former를 백본으로 사용함으로써 Track Query의 잠재력을 극대화하였다.

## 🛠️ Methodology

### 전체 시스템 구조

ROVIS의 구조는 Mask2Former를 기반으로 하며, 기본적으로 Backbone, Pixel Decoder, Transformer Decoder, 그리고 Class/Mask 예측 헤드로 구성된다. 전체 파이프라인은 다음과 같은 흐름으로 동작한다.

1. **초기 프레임(Frame 0):** 고정된 정적 쿼리(Static Queries)를 사용하여 객체를 탐지하고 마스크를 예측한다.
2. **쿼리의 전이:** Transformer Decoder의 최종 단계에서 출력된 객체 쿼리(Object Queries) 중, 실제 객체로 예측된 쿼리들만 선택하여 다음 프레임의 추적 쿼리(Track Queries)로 전달한다.
3. **후속 프레임(Frame $t$):** 정적 쿼리(새로운 객체 탐지용)와 이전 프레임에서 전달된 추적 쿼리(기존 객체 유지용)를 결합하여 입력으로 사용한다.
4. **업데이트 및 종료:** 추적 쿼리는 매 프레임 업데이트되며, 추적하던 객체가 사라지면 모델은 해당 쿼리에 대해 "배경(Background)" 클래스 라벨을 예측하게 된다.

### 훈련 목표 및 손실 함수

본 모델은 완전 지도 학습(Fully supervised) 방식으로 훈련되며, 정답(Ground-truth) 인스턴스 ID를 활용한다. 훈련 시에는 무작위로 샘플링된 인접한 두 프레임($x_0, x_1$) 쌍을 사용하여 학습한다.

마스크 예측을 위해 Focal Loss와 Dice Loss를 결합한 $\mathcal{L}_{mask}$를 사용하고, 클래스 예측을 위해 Cross-Entropy Loss인 $\mathcal{L}_{cls}$를 사용한다.

$$\mathcal{L}_{mask} = \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

전체 손실 함수는 다음과 같이 정의되며, 각 프레임의 정적 쿼리에 의한 예측과 추적 쿼리에 의한 예측 손실을 모두 합산한다.

$$\mathcal{L} = \lambda_{cls}(\mathcal{L}^0_{cls} + \mathcal{L}^{track}_{cls} + \mathcal{L}^1_{cls}) + \lambda_{mask}(\mathcal{L}^0_{mask} + \mathcal{L}^{track}_{mask} + \mathcal{L}^1_{mask})$$

여기서 $\lambda_{cls} = 2.0, \lambda_{mask} = 5.0$으로 설정되었다.

### 학습 및 추론 절차

* **학습 절차:** 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 예측값과 정답 인스턴스를 매칭한다. 특히 추적 쿼리는 이전 프레임의 동일한 인스턴스를 예측하도록 강제된다.
* **쿼리 증강(Query Augmentation):** 모델의 강건성을 위해 추적 쿼리가 누락되는 상황(False Negative)과 배경 쿼리가 추적 쿼리로 추가되는 상황(False Positive)을 확률적으로 시뮬레이션하여 학습시킨다.
* **추론 절차:** 객체가 일시적으로 사라졌을 때 즉시 쿼리를 제거하지 않고 $\Delta t=9$ 프레임 동안 유지하는 '비활성 내성(Inactivity tolerance)' 전략을 사용한다. 또한, 중복 예측을 방지하기 위해 Matrix NMS를 적용한다.

## 📊 Results

### 실험 설정

* **데이터셋:** YouTube-VIS 2019 (짧은 비디오), OVIS (심한 가려짐), UVO (오픈 월드, 긴 비디오)
* **백본:** ResNet50 및 Swin-L
* **지표:** Average Precision (AP) 및 Average Recall (AR)

### 주요 결과

1. **YouTube-VIS 2019:** ROVIS(ResNet50)는 45.5 AP를 기록하여, 최상위 오프라인 모델인 Mask2Former VIS(46.4 AP)에 근접하는 성능을 보였다. 이는 온라인 방식임에도 오프라인 방식의 정확도에 도달할 수 있음을 의미한다.
2. **OVIS:** 가려짐이 심한 이 데이터셋에서 ROVIS는 Mask2Former+IoU 베이스라인보다 10 AP 포인트 높은 30.2 AP를 달성하였다. 특히 Swin-L 백본 사용 시 42.6 AP로 매우 높은 성능을 보였으며, 이는 IDOL과 대등한 수준이다.
3. **UVO:** 오픈 월드 환경의 UVO 데이터셋에서 ROVIS는 다른 모든 모델을 압도하였다. 특히 Swin-L 모델은 32.7 AP를 기록하며, Mask2Former VIS(27.3 AP)나 IDOL(16.8 AP, ResNet50 기준)보다 훨씬 강건한 추적 능력을 보여주었다.
4. **효율성:** ResNet50 기반 모델은 12 fps, Swin-L 기반 모델은 5 fps의 추론 속도를 기록하여 실시간 적용 가능성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 강력한 이미지 분할 모델의 성능을 비디오 도메인으로 확장하기 위해 복잡한 시공간 모델링 대신 단순한 쿼리 전이(Query propagation) 메커니즘을 사용한 것이 주효했다. 실험 결과, Track Query 아이디어는 백본 모델의 성능이 충분히 높을 때 그 잠재력이 완전히 발휘된다는 점이 확인되었다. (예: vanilla DETR 기반의 TrackFormer보다 Mask2Former 기반의 ROVIS가 훨씬 강력함)

또한, 오프라인 모델들이 메모리 한계로 인해 긴 비디오에서 성능이 급격히 저하되는 반면, ROVIS는 온라인 방식으로 동작하므로 비디오 길이에 관계없이 일관된 성능을 유지할 수 있다는 점이 큰 강점이다. 다만, 모델이 여전히 정답 ID에 의존하여 훈련된다는 점과, 추적 쿼리의 누적된 오류가 전파될 가능성이 있다는 점은 향후 해결해야 할 과제로 보인다.

## 📌 TL;DR

ROVIS는 Mask2Former의 강력한 분할 능력과 TrackFormer의 효율적인 쿼리 전이 메커니즘을 결합한 완전 온라인 VIS 모델이다. 고비용의 시공간 어텐션 없이도 추적 쿼리만으로 객체의 정체성을 유지함으로써, GPU 메모리 제약 없이 긴 비디오와 복잡한 가려짐 상황을 효과적으로 처리할 수 있다. 특히 오픈 월드 데이터셋(UVO)에서 기존 모델들을 압도하는 성능을 보여주었으며, 이는 향후 효율적인 실시간 비디오 인스턴스 분할 연구의 중요한 기점이 될 가능성이 크다.
