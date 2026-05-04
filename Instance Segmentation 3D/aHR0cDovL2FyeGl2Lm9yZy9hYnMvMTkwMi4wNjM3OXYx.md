# PointIT: A Fast Tracking Framework Based on 3D Instance Segmentation

Yuan Wang, Yang Yu, Ming Liu (2019)

## 🧩 Problem to Solve

최근의 객체 추적(Object Tracking) 프레임워크들은 주로 2D 이미지 시퀀스에 집중되어 있으며, 3D 포인트 클라우드(Point Cloud) 상에서 객체를 추적하는 연구는 상대적으로 부족하다. 기존의 2D 추적 방식이나 3D Bounding Box 기반의 추적 방식은 주로 Intersection-over-Union (IoU) 매칭 행렬을 사용하여 연속된 프레임 간의 객체를 연결한다. 그러나 3D 객체 검출 모델들은 객체의 방향(Orientation) 예측에서 오차가 발생하는 경우가 많으며, 이는 IoU 행렬의 정확도를 떨어뜨려 객체 간의 상호작용(Interaction) 문제나 ID 전환(Identity Switch) 문제를 야기한다.

본 논문의 목표는 3D LiDAR 데이터를 효율적으로 처리하여 실시간성에 가까운 속도를 확보하면서도, 3D 공간 정보를 활용해 정밀한 객체 추적이 가능한 PointIT 프레임워크를 제안하는 것이다. 특히 3D 인스턴스 세그멘테이션(Instance Segmentation)을 통해 객체의 위치를 정밀하게 파악하고, 이를 확장된 SORT 알고리즘과 결합하여 추적 성능을 높이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 포인트 클라우드를 2D 구면 이미지(Spherical Image)로 투영하여 연산 효율성을 높이고, 가벼운 네트워크 구조를 통해 실시간 3D 인스턴스 세그멘테이션을 수행하는 것이다.

중심적인 기여 사항은 다음과 같다.

1. **효율적인 데이터 표현**: 3D 포인트 클라우드를 $64 \times 512 \times 4$ 크기의 구면 이미지로 변환하여 2D 컨볼루션 신경망을 적용할 수 있게 함으로써 연산 속도를 향상시켰다.
2. **경량화된 인스턴스 세그멘테이션**: Mask R-CNN 구조를 채택하되, 무거운 ResNet 대신 MobileNet을 Backbone으로 사용하여 계산 복잡도를 줄였다.
3. **Extended SORT 알고리즘**: 기존 SORT의 IoU 기반 매칭 방식에 3D 공간상의 정규화된 거리(Normalized Distance) 정보를 추가한 비용 함수를 도입하여, 객체 간의 가려짐이나 교차 상황에서도 안정적인 추적이 가능하도록 개선하였다.

## 📎 Related Works

논문에서는 인스턴스 세그멘테이션, 다중 객체 추적(MOT), 데이터 퓨전 세 가지 관점에서 관련 연구를 설명한다.

1. **Instance Segmentation**: 기존의 MNC와 같은 다단계 모델은 정확도는 높으나 연산 시간이 너무 길어 자율주행 차량에 적용하기 어렵다. 또한 DeepMask나 FCIS 같은 FCN 기반 방식은 인접한 인스턴스가 겹칠 때 마스크 예측 오류가 발생하는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 Mask R-CNN을 기반으로 하되, 구면 이미지 투영 방식을 도입하여 메모리 효율성을 높였다.
2. **Multiple Object Tracking**: SORT는 칼만 필터와 헝가리안 알고리즘을 사용해 실시간성을 확보했지만, 객체 간 교차 시 ID 전환이 빈번하다는 단점이 있다. DeepSORT는 이를 해결하기 위해 딥러닝 기반의 외형 특징(Appearance Feature)을 사용하지만, 특정 데이터셋에 의존적인 학습이 필요하여 범용성이 떨어진다.
3. **Data Fusion**: MV3D나 VoxelNet과 같이 다양한 뷰를 결합하거나 복셀(Voxel) 단위를 사용하는 방식이 제안되었으나, 본 논문은 SqueezeSeg나 PointSeg와 같이 구면 좌표계로의 투영 방식을 채택하여 효율성을 극대화하였다.

## 🛠️ Methodology

### 1. Network Input: 3D to Spherical Projection

3D LiDAR 데이터를 처리하기 위해, 본 연구는 데이터를 구면 좌표계로 투영하여 이미지 형태로 변환한다. 입력 데이터는 각 포인트의 데카르트 좌표 $(x, y, z)$와 반사도(Reflectivity)를 포함하는 4개 채널로 구성된다. 투영 과정은 다음과 같은 방정식으로 정의된다.

방위각(Azimuth angle) $\alpha$와 천정각(Zenith angle) $\beta$는 다음과 같이 계산된다.
$$\alpha = \arcsin\left(\frac{z}{\sqrt{x^2+y^2+z^2}}\right), \quad \bar{\alpha} = \lfloor \frac{\alpha}{\Delta\alpha} \rfloor$$
$$\beta = \arcsin\left(\frac{y}{\sqrt{x^2+y^2}}\right), \quad \bar{\beta} = \lfloor \frac{\beta}{\Delta\beta} \rfloor$$

여기서 $\Delta\alpha$와 $\Delta\beta$는 생성할 구면 이미지의 크기를 결정하며, 본 논문에서는 $64 \times 512$로 설정하였다. 결과적으로 3D 포인트 클라우드는 $64 \times 512 \times 4$ 형태의 텐서로 변환되어 모델의 입력으로 사용된다.

### 2. Instance Segmentation Network

실시간 성능을 위해 MobileNet을 Backbone으로 사용하는 Mask R-CNN 구조를 설계하였다.

- **Encoder**: MobileNet의 핵심인 Depthwise Separable Convolution 블록을 사용하여 파라미터 수를 줄였다. 총 4번의 다운샘플링을 거쳐 최종 특징 맵의 크기는 $H/16 \times W/16 \times 512$가 된다.
- **FPN & RPN**: Feature Pyramid Networks (FPN)를 통해 다양한 스케일의 특징을 추출하고, Region Proposal Network (RPN)가 객체의 후보 영역(ROI)을 생성한다.
- **Heads**: 생성된 ROI를 바탕으로 객체의 클래스를 분류하고 Bounding Box를 예측하는 Classifier graph와, $28 \times 28$ 크기의 픽셀 단위 마스크를 생성하는 Mask graph가 병렬적으로 작동한다.

### 3. Extended SORT

추적 단계에서는 기존 SORT 알고리즘을 3D 공간 정보가 반영되도록 확장하였다.

**상태 추정(State Estimation)**:
객체의 상태는 2D Bounding Box 정보 $\mathbf{X}_{\text{Object}} = (x^p, y^p, s, r, \bar{x}, \bar{y}, \bar{s})^\top$와 3D 공간상의 중심 좌표 및 속도 정보 $\mathbf{X}_{\text{Center}} = (x^w, y^w, z^w, \bar{V}_x, \bar{V}_y, \bar{V}_z, \bar{A}_x, \bar{A}_y)^\top$로 정의하여 칼만 필터를 통해 예측한다.

**데이터 연관(Data Association)**:
두 프레임 간의 객체를 매칭하기 위해 IoU와 3D 거리를 결합한 비용 함수를 사용한다.
$$\text{Graph}(i, j) = \alpha \cdot I(i, j) + \beta \cdot D(i, j)$$
여기서 $I(i, j)$는 Bounding Box 간의 IoU이며, $D(i, j)$는 다음과 같이 정의된 거리 기반 가중치이다.
$$D(i, j) = \exp[-\text{dis}(i, j)], \quad \text{dis} = \frac{2}{\sqrt{\|P_i - P_j\|}}, \quad P = [x, y, z]^\top$$
$\alpha$와 $\beta$는 각 지표의 가중치를 조절하며, 본 논문에서는 $\alpha = 0.5, \beta = 0.5$로 설정하여 동일한 비중을 두었다. 이후 헝가리안 알고리즘을 통해 전체 비용을 최소화하는 매칭을 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋**: KITTI 3D Object Track 데이터셋을 구면 이미지로 변환하여 사용하였다.
- **평가 지표**: MOTA (Accuracy), MOTP (Precision), MT (Mostly Tracked), ML (Mostly Lost), IDsw (ID Switches), FM (False Negatives), FP (False Positives) 등을 사용하였다.

### 주요 결과

1. **인스턴스 세그멘테이션 성능**: MobileNet 기반 모델은 $AP_{0.5} = 0.617$을 달성하였다. ResNet50 기반 모델과 비교했을 때 AP는 약 4% 낮았으나(ResNet의 경우 0.66), 런타임은 0.091초에서 0.061초로 크게 단축되어 효율성을 입증하였다.
2. **추적 성능 (PointIT vs. SORT)**:
   - MOTA: SORT(0.451) $\rightarrow$ PointIT(0.457)로 소폭 상승하였다.
   - IDsw: 3D 공간 정보의 도입으로 인해 ID 전환 횟수가 줄어들었으며, 특히 MT(Mostly Tracked) 지표가 0.137에서 0.155로 향상되었다.
   - 전체 파이프라인의 처리 속도는 약 15 fps를 기록하였다.

## 🧠 Insights & Discussion

본 연구는 3D 포인트 클라우드를 2D 구면 이미지로 투영하여 처리함으로써, 3D 데이터가 갖는 막대한 연산량 문제를 해결하고 기존의 효율적인 2D 딥러닝 아키텍처(MobileNet, Mask R-CNN)를 활용할 수 있음을 보여주었다. 특히 단순한 IoU 기반 매칭에서 벗어나 3D 중심점 거리를 비용 함수에 추가함으로써, 객체 간의 상호작용이나 일시적인 가려짐 상황에서도 추적의 안정성을 높였다는 점이 긍정적이다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, ResNet과 MobileNet의 성능 차이가 예상보다 작았는데($AP$ 차이 4%), 이는 구면 이미지의 도메인 분포 특성상 네트워크가 깊어지는 것이 반드시 성능 향상으로 이어지지 않는 상한선(Upper bound)이 존재할 가능성을 시사한다. 둘째, LiDAR 센서의 특성상 물체의 재질이나 강도에 따라 포인트 유실이 발생하며, 이로 인해 예측된 마스크의 형태가 불규칙해지는 현상이 관찰되었다. 이는 향후 데이터 전처리나 정제 단계에서 해결해야 할 과제로 보인다.

## 📌 TL;DR

PointIT은 3D LiDAR 데이터를 구면 이미지로 투영하고, 경량화된 MobileNet-Mask R-CNN 구조를 통해 실시간으로 3D 인스턴스 세그멘테이션을 수행하는 추적 프레임워크이다. 기존 SORT 알고리즘에 3D 공간 거리 정보를 결합하여 ID 전환 문제를 완화하였으며, 연산 효율성과 추적 정확도 사이의 균형을 맞추었다. 이 연구는 자율주행 시스템에서 3D 포인트 클라우드를 이용한 저지연·고정밀 객체 추적을 구현하는 데 중요한 방법론을 제시한다.
