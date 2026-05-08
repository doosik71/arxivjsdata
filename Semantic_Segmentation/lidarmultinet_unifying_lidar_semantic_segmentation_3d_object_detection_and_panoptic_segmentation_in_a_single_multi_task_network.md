# LidarMultiNet: Unifying LiDAR Semantic Segmentation, 3D Object Detection, and Panoptic Segmentation in a Single Multi-task Network

Dongqiangzi Ye, Weijia Chen, Zixiang Zhou, Yufei Xie, Yu Wang, Panqu Wang, Hassan Foroosh (2022)

## 🧩 Problem to Solve

자율주행의 핵심 기술인 LiDAR 3D semantic segmentation은 포인트 클라우드의 대규모 특성과 희소성(sparsity)으로 인해 2D 이미지나 실내 3D 세그멘테이션 방법론을 직접 적용하기 어렵다는 문제점이 있다. 특히 기존의 Voxel 기반 LiDAR 세그멘테이션 네트워크들은 Sparse Convolution을 사용하고 Encoder-Decoder 구조를 채택하지만, 이로 인해 글로벌 컨텍스트 정보(global contextual information)를 학습하는 데 한계가 있다.

또한, 3D semantic segmentation, 3D object detection, panoptic segmentation과 같은 주요 LiDAR 인지 작업들이 서로 독립적인 별도의 네트워크에서 수행되어 왔다. 이는 각 작업 간의 시너지 효과를 활용하지 못하게 하며, 특히 panoptic segmentation의 경우 검출(detection)과 세그멘테이션(segmentation) 모델을 독립적으로 결합했을 때의 성능이 엔드투엔드(end-to-end) 방식보다 높게 나타나는 불일치 문제가 존재한다. 본 논문의 목표는 이러한 세 가지 주요 작업을 하나의 통합된 네트워크에서 수행하여 성능을 극대화하는 LidarMultiNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 강력한 Voxel 기반의 Encoder-Decoder 네트워크를 중심으로, 전역적 특징을 추출하는 Global Context Pooling (GCP) 모듈과 다중 작업 학습(Multi-task Learning) 구조를 통합하는 것이다.

가장 중점적인 기여는 Submanifold Convolution의 한계인 좁은 수용 영역(receptive field) 문제를 해결하기 위해, 3D Sparse Tensor를 2D Dense BEV(Bird's Eye View) 맵으로 투영하여 전역적 컨텍스트를 학습하고 이를 다시 3D 공간으로 되돌리는 GCP 모듈을 설계한 점이다. 또한, 3D object detection과 BEV segmentation을 보조 작업(auxiliary tasks)으로 추가하여 세그멘테이션 성능을 향상시키는 시너지를 창출하였다. 마지막으로, 'Thing' 클래스(차량, 보행자 등)의 공간적 일관성을 높이기 위해 검출된 Bounding Box 정보를 활용하는 2단계 정제(2nd-stage refinement) 모듈을 도입하였다.

## 📎 Related Works

기존의 LiDAR semantic segmentation 연구들은 주로 포인트 클라우드를 3D Voxel, 2D BEV, 또는 Range-view 맵으로 변환하여 처리하였다. PolarNet과 같은 방식은 2D polar BEV 맵을 사용하여 포인트 분포의 균형을 맞추었으며, 최근에는 3D Sparse Convolution을 활용한 Voxel 기반 방식들이 주를 이루고 있다. 그러나 많은 최신 연구들이 세부적인 기하학적 관계(fine-grained details)를 복원하는 데 집중하는 반면, 본 연구는 Voxel 기반 네트워크에서 부족한 전역적 특징 학습을 강화하는 데 집중한다.

Panoptic segmentation 분야에서는 주로 bottom-up 방식이 사용되었으며, 검출 네트워크에서 높이 정보의 손실로 인해 세그멘테이션 작업으로의 전이가 어려웠다. 이로 인해 최적의 세그멘테이션 모델과 최적의 검출 모델의 설계가 서로 호환되지 않는 문제가 있었으며, 본 논문은 이를 해결하기 위해 세 작업을 단일 네트워크에서 동시에 학습시키는 통합 구조를 제안하며 기존의 독립적 결합 방식과 차별화를 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

LidarMultiNet은 3D Sparse Convolution 기반의 Encoder-Decoder 구조를 기본 뼈대로 하며, Encoder와 Decoder 사이에 GCP 모듈이 위치한다. Decoder의 출력은 3D Segmentation Head로 연결되어 포인트별 라벨을 예측하며, GCP 모듈에서 생성된 2D BEV 특징 맵은 3D Detection Head와 BEV Segmentation Head로 연결되어 보조 작업을 수행한다.

### 주요 구성 요소 및 절차

1. **Voxelization**: 입력 포인트 클라우드 $P$를 Voxel 인덱스 $v_i$로 변환한다. Voxel 특징 $V_j$는 MLP와 max-pooling을 통해 다음과 같이 생성된다.
   $$V_j = \max_{v_i=v_j}(\text{MLP}(p_i)), \quad j \in (1 \dots M)$$
   여기서 $M$은 유니크한 Voxel 인덱스의 수이다.

2. **Sparse Voxel-based Encoder-Decoder**: 3D UNet 구조를 채택하여 Encoder에서 특징을 1/8 크기로 다운샘플링하고 Decoder에서 다시 업샘플링한다. 효율성을 위해 Encoder와 Decoder 간에 Key indices를 공유하며, Skip-connection을 통해 세부 특징을 보존한다.

3. **Global Context Pooling (GCP)**: Submanifold Convolution은 유효한 Voxel에서만 연산이 일어나므로 전역 특징 학습이 어렵다. 이를 해결하기 위해 다음 과정을 거친다.
   - 3D Sparse Tensor를 2D Dense BEV 특징 맵 $F_{bev}$로 투영한다 (높이 차원 $\text{concat}$).
   - 2D multi-scale CNN을 통해 전역 컨텍스트를 추출한다.
   - 추출된 특징을 다시 Dense Voxel 맵으로 reshape한 후, Sparse Voxel 특징으로 변환하여 Decoder에 전달한다.

4. **Multi-task Learning 및 손실 함수**:
   - **Segmentation**: Cross-entropy loss와 Lovasz loss를 결합하여 $\mathcal{L}_{SEG} = \mathcal{L}_{ce} + \mathcal{L}_{Lovasz}$로 정의한다.
   - **Detection**: CenterPoint 헤드를 사용하여 Focal loss($\mathcal{L}_{hm}$)와 $L1$ loss($\mathcal{L}_{reg}, \mathcal{L}_{iou}$)를 사용한다.
   - **BEV Segmentation**: $\mathcal{L}_{BEV}$를 통해 보조 손실을 제공한다.
   - **Total Loss**: 각 작업의 불확실성(uncertainty) $\sigma_i$를 학습 파라미터로 사용하는 가중치 합으로 정의한다.
     $$\mathcal{L}_{total} = \sum_{i \in \{SEG,DET,BEV\}} \frac{1}{2\sigma_i^2}\mathcal{L}_i + \frac{1}{2}\log\sigma_i^2$$

5. **Second-stage Refinement**: 1단계 예측의 공간적 불일치를 해결하기 위해, 검출된 Bounding Box 내부의 포인트를 대상으로 수행한다. 포인트의 로컬 좌표($P_{local}$)와 Voxel 특징, BEV 특징을 결합하여 포인트별 마스크 점수($S_{point}$)와 박스별 클래스 점수($S_{box}$)를 예측하며, 이를 1단계 결과와 융합하여 최종 결과를 도출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Waymo Open Dataset (WOD)을 사용하였으며, 총 23개의 클래스를 대상으로 한다.
- **학습 세부사항**: AdamW 옵티마이저와 One-cycle learning rate policy를 사용하였고, 8개의 V100 GPU에서 학습하였다. 입력으로 현재 프레임과 과거 2개 프레임을 합쳐 더 밀집된 포인트 클라우드를 구성하였다.

### 정량적 결과

LidarMultiNet은 Waymo 3D semantic segmentation 챌린지 2022에서 **mIoU 71.13**을 기록하며 1위를 차지하였다. 특히 전체 22개 클래스 중 15개 클래스에서 가장 높은 정확도를 보였다.

### Ablation Study

검증 세트(validation set)에서의 분석 결과, 기본 네트워크(mIoU 69.90) 대비 다음과 같은 성능 향상이 확인되었다.

- **Multi-frame 입력**: +0.59 mIoU
- **GCP 모듈**: +0.94 mIoU
- **보조 손실(BEV Seg, Detection)**: +0.63 mIoU
- **2단계 정제(2nd-stage)**: +0.34 mIoU
최종적으로 단일 모델 기준 mIoU 72.40을 달성하였으며, TTA(Test-Time Augmentation)와 앙상블을 적용했을 때는 최대 73.78까지 성능이 향상되었다.

## 🧠 Insights & Discussion

본 논문은 LiDAR 인지 작업에서 전역적 컨텍스트의 중요성을 입증하였다. 특히 3D Sparse Convolution의 효율성을 유지하면서도, 2D BEV 투영을 통한 GCP 모듈을 통해 수용 영역을 획기적으로 넓힌 점이 성능 향상의 핵심 요인으로 분석된다. 또한, 세그멘테이션, 검출, 파놉틱 세그멘테이션을 하나의 네트워크로 통합함으로써 각 작업이 서로의 학습을 돕는 시너지 효과를 증명하였다.

다만, 본 보고서의 주된 성과가 Waymo 챌린지의 세그멘테이션 리더보드에 집중되어 있다는 점이 한계이다. 저자들 또한 결론에서 언급하였듯이, 통합 네트워크가 3D object detection이나 panoptic segmentation의 개별 벤치마크에서도 독립적인 최신 모델들보다 우수한 성능을 내는지에 대한 추가 검증이 필요하다. 또한 2단계 정제 과정이 연산 비용을 증가시킬 수 있으므로, 실시간성 측면에서의 분석이 보완되어야 할 것으로 보인다.

## 📌 TL;DR

LidarMultiNet은 3D semantic segmentation, object detection, panoptic segmentation을 하나의 프레임워크로 통합한 네트워크이다. 3D Sparse Convolution의 한계를 극복하기 위해 BEV 맵 기반의 **Global Context Pooling (GCP)** 모듈을 제안하고, 다중 작업 학습 및 2단계 정제 과정을 통해 Waymo 챌린지 1위를 달성하였다. 이 연구는 LiDAR 기반 인지 작업들이 서로 독립적이지 않고 상호 보완적일 수 있음을 보여주었으며, 향후 통합 인지 모델 연구에 중요한 이정표가 될 가능성이 크다.
