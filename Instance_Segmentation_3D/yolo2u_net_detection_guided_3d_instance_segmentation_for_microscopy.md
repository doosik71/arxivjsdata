# YOLO2U-NET: DETECTION-GUIDED 3D INSTANCE SEGMENTATION FOR MICROSCOPY

Amirkoushyar Ziabari, Derek C. Rose, Abbas Shirinifard, David Solecki (2022)

## 🧩 Problem to Solve

본 논문은 현미경 이미지(Microscopy imaging) 내의 세포 핵(Cell nuclei)에 대한 정확한 3D Instance Segmentation을 수행하는 것을 목표로 한다.

일반적으로 현미경 이미지는 2D 투영 이미지들을 쌓아서 3D 시각화를 구현하는데, 이 과정에서 평면 외 여기(out-of-plane excitation) 문제나 z-축의 낮은 해상도 문제가 발생한다. 이로 인해 실제로는 겹치지 않는 세포들이 이미지 상에서는 겹쳐 보일 수 있으며, 이는 전문가조차 개별 세포를 식별하는 데 어려움을 겪게 만든다. 또한, 생의학 이미지 데이터는 객체의 방향과 농도가 무작위적이고, 경계가 불분명하며, 노이즈가 심해 자동화된 분할(Segmentation)이 매우 까다롭다.

따라서 본 연구의 목적은 이러한 3D 볼륨 데이터에서 개별 세포를 정확하게 검출하고 분리해낼 수 있는 포괄적인 3D Instance Segmentation 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"2D 검출(Detection)을 통한 3D 위치 추정 후, 국소 영역에 대한 3D 분할(Segmentation)을 수행한다"**는 것이다.

구체적으로는 YOLO라는 효율적인 2D Object Detection 네트워크와 U-Net이라는 강력한 Segmentation 네트워크를 결합하였다. 3D 전체 볼륨에 대해 직접적으로 3D Convolution을 수행하는 것은 계산 비용이 기하급수적으로 증가하므로, 세 가지 직교 평면(Orthogonal planes)에서 2D 검출을 수행하고 이를 융합하여 3D Bounding Box를 생성하는 **2.5D Fusion 알고리즘**을 도입하였다. 이후, 생성된 3D Bounding Box 내부의 주 세포(Primary cell)만을 분할하도록 3D U-Net을 설계함으로써 계산 효율성과 정확도를 동시에 확보하였다.

## 📎 Related Works

기존의 3D Segmentation 접근 방식들은 다음과 같은 한계점을 가진다:

1. **Fully Convolutional Networks (FCNs):** 주로 2D 이미지에 최적화되어 있으며, 3D 전체 볼륨에 적용하기에는 계산 비용이 너무 크다.
2. **DeepSynth:** 합성 데이터를 사용하여 3D U-Net을 학습시키지만, 인접한 세포를 분리하기 위해 Watershed 알고리즘과 형태학적 후처리(Morphological post-processing)에 의존한다. 이는 과분할(Over-segmentation) 문제를 야기할 수 있다.
3. **DeepCell:** 딥러닝 기반의 Watershed 접근 방식을 통해 겹치는 세포를 처리하지만, 여전히 복잡한 볼륨에서는 한계가 있다.
4. **StarDist:** 3D 분할이 가능하지만, 객체의 모양이 Star-convex 형태여야 한다는 제약이 있다.

본 연구의 YOLO2U-Net은 후처리 단계의 Watershed나 형태학적 연산을 제거하고, Detection-guided 방식으로 각 객체를 개별적으로 분할함으로써 기존 방식의 고질적인 문제인 과분할 및 세포 경계 소실 문제를 해결하고자 한다.

## 🛠️ Methodology

YOLO2U-Net의 전체 파이프라인은 **[2D Detection $\rightarrow$ 2.5D Fusion $\rightarrow$ 3D Segmentation]**의 세 단계로 구성된다.

### 1. 2.5D YOLO-based Fusion Algorithm

3D 볼륨의 계산 복잡도를 줄이기 위해 2D 직교 평면(X-Y, X-Z, Y-Z)에서 세포를 검출한다.

- **2D Detection:** Modified YOLOv2를 사용하여 각 평면에서 2D Bounding Box를 예측한다. 이때 신뢰도(Confidence)가 $50\%$ 이상인 박스만 유지하며, Non-Maximum Suppression(NMS)을 통해 중복을 제거한다.
- **Fusion Process:** 서로 다른 뷰에서 검출된 2D 박스들을 쌍으로 비교하여 교차 영역을 찾는다. 이를 통해 3D 좌표 $[x_{min}, x_{max}, y_{min}, y_{max}, z_{min}, z_{max}]$를 추정한다.
- **Clustering & Refinement:** 중첩도가 $5\%$ 이상인 3D 제안 박스들을 클러스터링하고, 각 클러스터 내 좌표들의 중앙값(Median)을 계산하여 최종적인 하나의 3D Bounding Box를 결정한다. 마지막으로 다시 한번 NMS를 적용해 중복을 방지한다.

### 2. 3D U-Net for Guided Segmentation

추정된 3D Bounding Box 내부의 데이터를 3D U-Net의 입력으로 사용한다.

- **입력 정규화:** 세포의 크기가 다양하므로, 모든 3D Bounding Box를 $48 \times 48 \times 48$ 크기의 고정된 큐브 형태로 리스케일링(Rescaling)한다. 크기가 작은 경우 zero-padding을 적용한다.
- **분할 목표:** 네트워크는 해당 큐브 내의 '주 세포(Primary cell)'와 나머지 배경(인접 세포의 일부 포함)을 구분하도록 학습된다.
- **복원 및 통합:** 분할이 완료된 결과물은 다시 원래의 크기로 되돌린 후, 원본 볼륨의 위치에 배치한다. 최종적으로 각 위치에 대해 $\text{argmax}$ 연산을 수행하여 각 세포에 고유 라벨을 부여함으로써 3D Instance Segmentation을 완성한다.

## 📊 Results

### 실험 설정

- **데이터셋:** CompuCell3D 툴킷을 사용하여 합성된 뇌 조직 세포 데이터셋을 사용하였다. 실제 현미경 데이터의 특성을 모사하기 위해 3D Gaussian Blur(점 확산 함수近似)와 Gaussian Noise를 추가하였다.
- **비교 대상:** DeepCell, DeepSynth, Two-Tier CNN.
- **평가 지표:**
  - Intersection-over-Union (IoU): $IoU = \frac{Cell_{target} \cap Cell_{predicted}}{Cell_{target} \cup Cell_{predicted}}$
  - Precision ($P$), Recall ($R$), Jaccard Index ($J$)를 IoU 임계값($th$)에 따라 계산하며, 이를 통합하여 mAP, mAR, mAJ를 산출한다.

### 주요 결과

- **정량적 성능:** Table 1에 따르면 YOLO2U-Net은 mAP($0.367$), mAR($0.39$), mAJ($0.263$)에서 모든 비교 대상 모델보다 높은 성능을 보였다.
- **IoU 임계값 분석:** 특히 IoU가 $0.7$ 이상인 높은 정밀도 영역에서 YOLO2U-Net이 다른 모델들을 압도하였다. 이는 사전 3D Localization 단계가 분할 정밀도를 크게 향상시켰음을 시사한다.
- **정성적 분석:** Figure 4의 결과에서 DeepSynth는 Watershed 후처리로 인한 과분할이 발생하고, DeepCell은 경계 분리가 불분명한 반면, YOLO2U-Net은 모호한 경계에서도 세포를 정확하게 분리해내는 모습을 보였다.

### Ablation Study

- **Baseline 1 (Perfect 3D BBs):** 3D Bounding Box가 완벽하게 주어졌을 때의 성능을 측정하여 3D U-Net의 잠재적 한계를 분석하였다.
- **Baseline 2 (Perfect 2D BBs):** 2D 박스는 완벽하지만 Fusion 알고리즘만 사용했을 때의 성능을 측정하였다.
- **결과:** 데이터셋의 복잡도가 낮을 때는 YOLOv2와 Fusion 알고리즘의 개선만으로도 성능을 거의 완벽하게 끌어올릴 수 있으나, 매우 복잡한 데이터셋(Dataset 3)에서는 완벽한 Bounding Box가 주어져도 분할 오류가 발생하였다. 이는 3D U-Net 자체의 구조적 개선이 필요함을 의미한다.

## 🧠 Insights & Discussion

**강점:**

- **계산 효율성:** 전체 볼륨이 아닌 국소적인 Bounding Box 내부만 처리하므로 연산량이 크게 줄어든다.
- **유연성:** 입력 리스케일링을 통해 다양한 크기의 세포에 적응적이며, 모듈형 구조(YOLO $\rightarrow$ Fusion $\rightarrow$ U-Net)이므로 각 구성 요소를 최신 모델로 쉽게 교체할 수 있다.
- **후처리 제거:** Watershed와 같은 전통적인 후처리를 제거하여 과분할 문제를 원천적으로 차단하였다.

**한계 및 논의:**

- **모델 의존성:** Ablation Study 결과, 전체 성능이 YOLOv2의 검출 성능과 Fusion 알고리즘의 정확도에 강하게 의존한다. 즉, Detection 단계에서 누락된 세포는 이후 단계에서 복구될 수 없다.
- **구조적 한계:** 매우 복잡한 실제 데이터 환경에서는 단순한 3D U-Net만으로는 분할 정밀도를 높이는 데 한계가 있음이 드러났다. 저자들은 이를 해결하기 위해 Mixed Scale Dense Network 등의 대안을 제시하였다.

## 📌 TL;DR

본 논문은 3D 현미경 이미지의 세포 분할을 위해 **2D Detection(YOLOv2) $\rightarrow$ 2.5D Fusion $\rightarrow$ 3D Segmentation(U-Net)**으로 이어지는 단계적 프레임워크인 **YOLO2U-Net**을 제안하였다. 이 방법은 3D 전체 볼륨을 직접 처리하는 대신, 세 방향의 2D 뷰를 융합해 객체의 위치를 먼저 찾고 해당 영역만 정밀 분할함으로써 계산 효율성과 정확도를 모두 높였다. 실험 결과, 기존의 3D Instance Segmentation 방법들보다 특히 높은 IoU 영역에서 우수한 성능을 보였으며, 향후 end-to-end 학습 모델로의 확장 가능성을 제시하였다.
