# Real-time Surgical Tools Recognition in Total Knee Arthroplasty Using Deep Neural Networks

Moazzem Hossain, Soichi Nishio, Takafumi Hiranaka, and Syoji Kobashi (2017)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 문제는 인공 무릎 관절 치환술(Total Knee Arthroplasty, 이하 TKA) 과정에서 사용되는 수많은 수술 도구들을 실시간으로 정확하게 인식하는 것이다. TKA는 무릎 관절염 환자의 통증 완화와 기능 개선을 위해 널리 시행되는 수술이지만, 수술 단계별로 사용하는 도구가 매우 다양하여 절차가 복잡하다. 실제로 전체 수술 과정에는 약 27개의 절차가 포함되며, 약 120종의 서로 다른 카테고리 도구들이 사용된다.

이러한 복잡성으로 인해 수술 집도의나 보조자가 수술 도구를 수동으로 식별하는 데 어려움이 있으며, 이는 수술 효율성에 영향을 줄 수 있다. 또한, 수술 도구의 존재 여부와 움직임을 실시간으로 파악하는 것은 수술의 운영 단계(operational phase)를 인식하고 전체적인 수술 워크플로우를 식별하는 데 필수적인 정보가 된다. 따라서 본 논문의 목표는 Convolutional Neural Network (CNN)를 이용하여 TKA 수술 중 도구를 실시간으로 인식하는 시스템을 개발함으로써, 스마트 글래스를 통해 집도의에게 필수 정보를 제공하고 수술 절차의 복잡성을 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실시간 수술 도구 탐지를 위해 Faster R-CNN 아키텍처를 TKA 수술 환경에 적용하고, 이를 통해 기존의 탐지 방법들보다 향상된 정확도와 속도를 달성한 점이다. 특히 스마트 글래스로부터 입력되는 비디오 프레임을 처리하여 실시간으로 도구의 Bounding Box와 클래스를 예측함으로써, 수술실 내에서 즉각적으로 활용 가능한 시스템의 가능성을 제시하였다. 또한, 제안된 방법이 향후 수술 단계 인식(operational phase recognition)을 위한 기초 데이터(baseline)로 활용될 수 있음을 시사한다.

## 📎 Related Works

기존의 수술 도구 탐지 연구는 크게 세 가지 방향으로 진행되어 왔다. 첫째는 Radiofrequency Identification (RFID) 태그를 이용한 방식, 둘째는 세그멘테이션(segmentation), 윤곽선 처리(contour processing) 및 3D 모델링 기반 방식, 셋째는 Viola-Jones 탐지 프레임워크를 이용한 방식이다. 최근에는 딥러닝 기반의 CNN 아키텍처가 컴퓨터 비전 분야에서 뛰어난 성능을 보이면서, R-CNN, Fast R-CNN, SSD, R-FCN 등의 모델들이 수술 도구 탐지 및 단계 인식에 활용되고 있다.

기존 연구들의 한계점은 많은 경우 수술 훈련 비디오를 사용한다는 점이다. 훈련용 데이터는 실제 수술 환경과 상당한 차이가 있어 실제 수술에서의 성능을 정확히 반영하지 못하는 경향이 있다. 본 연구는 이러한 한계를 극복하기 위해 실제 수술 중에 스마트 글래스로 촬영된 데이터를 사용하며, 단순한 존재 여부(presence detection) 확인을 넘어 도구의 공간적 경계(spatial bounds)를 탐지하는 Region-based CNN 방식을 채택하여 보다 종합적인 수술 품질 평가가 가능하도록 설계하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구에서 제안하는 시스템은 스마트 글래스를 통해 캡처된 TKA 수술 비디오 프레임을 입력으로 받아, Faster R-CNN 아키텍처를 통해 수술 도구를 실시간으로 탐지한다. 전체 파이프라인은 다음과 같은 단계로 구성된다.

1. **특징 추출 (Feature Extraction):** VGG-16 네트워크를 Backbone으로 사용하여 입력 이미지로부터 강력한 시각적 특징 맵(convolutional feature maps)을 추출한다.
2. **영역 제안 (Region Proposal):** 추출된 특징 맵 상에서 Region Proposal Network (RPN)가 슬라이딩 윈도우 방식을 통해 객체가 존재할 가능성이 높은 영역(object proposals)을 생성한다.
3. **분류 및 정제 (Classification and Refinement):** RPN에서 제안된 영역들의 특징을 풀링(pooling)하여 최종 분류 네트워크와 Bounding Box 정제 네트워크로 전달하며, 여기서 최종 도구 클래스와 정확한 좌표를 결정한다.

### 훈련 목표 및 손실 함수

RPN은 각 이미지에 대해 다음과 같은 손실 함수를 최적화하도록 학습된다.

$$ \mathcal{L}(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_{i} \text{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_{i^*} \text{reg}(t_i, t_i^*) $$

여기서 $i$는 입력 특징 맵의 슬라이딩 윈도우 각 위치에 해당하는 '앵커(anchor)'를 의미하며, $p_i$는 앵커의 객체성(Objectness) 확률, $t_i$는 예측된 Bounding Box의 좌표이다. $p_i^*$는 Ground-truth 라벨(Intersection over Union, IoU 기반)이며, $t_i^*$는 양성 앵커에 대응하는 Ground-truth 박스의 좌표이다. 손실 함수는 이진 객체성 라벨을 위한 분류 손실 $\mathcal{L}_{cls}$와 좌표 예측을 위한 회귀 손실 $\mathcal{L}_{reg}$의 가중치 조합으로 구성된다.

### 학습 절차 및 상세 설정

- **사전 학습 및 미세 조정 (Pre-training & Fine-tuning):** 일반적인 시각 특징을 학습하기 위해 ImageNet 데이터셋으로 네트워크를 사전 학습시킨 후, 라벨링된 TKA 수술 도구 데이터셋으로 미세 조정을 수행하였다.
- **앵커 라벨링:** Ground-truth 박스와의 IoU가 0.8 이상이면 양성(positive), 0.3 미만이면 음성(negative)으로 라벨링하였다.
- **최적화:** Stochastic Gradient Descent (SGD)를 사용하였으며, 40K iteration 동안 배치 크기 40, $3 \times 3$ 커널 크기로 최적화하였다.
- **데이터 증강:** 프레임을 무작위로 수평 뒤집기(horizontal flipping) 하여 데이터 양을 늘렸다.
- **검증 방법:** 데이터셋의 무작위성에 따른 영향을 줄이기 위해 Leave-One-Out Cross-Validation (LOOCV) 기법을 적용하였다. 이는 $n$개의 샘플 중 $n-1$개로 학습하고 1개로 검증하는 과정을 반복하여 평균 성능을 내는 방식이다.

## 📊 Results

### 실험 환경 및 데이터셋

- **데이터셋:** 2017년 일본 타카츠키 병원에서 촬영된 16개의 TKA 수술 비디오(25 fps, 프레임 크기 $654 \times 480$ 픽셀)를 사용하였다.
- **탐지 대상:** Femoral Drill Guide, Spherical Mill, Gap Gauge 등 총 31종의 수술 도구를 대상으로 하였으며, 도구당 약 900~1,200장의 어노테이션된 이미지를 사용하였다.
- **하드웨어:** 3.40 GHz CPU 및 NVIDIA GeForce 1080ti GPU 환경에서 cuDNN 라이브러리를 사용하여 실험을 진행하였다.

### 정량적 결과

제안 방법(Faster R-CNN 기반)을 기존의 State-of-the-art 방법들인 Fast R-CNN, DPM(Deformable Part Models)과 비교한 결과는 다음과 같다.

| 방법론 | mAP (Mean Average Precision) | 탐지 시간 (Detection Time) | 최소 정밀도 (Min) | 최대 정밀도 (Max) |
| :--- | :---: | :---: | :---: | :---: |
| **Proposed Method** | **87.6%** | **0.075 s** | **75** | **96** |
| Fast R-CNN | 84.48% | 0.159 s | 62 | 94 |
| DPM | 76.0% | 2.3 s | 57 | 84 |
| EdgeBox + Fast R-CNN | 20% | 0.134 s | 12 | 35 |

실험 결과, 제안된 방법이 mAP 87.6%로 가장 높은 정확도를 보였으며, 탐지 속도 또한 0.075초로 가장 빨라 실시간 처리가 가능함을 입증하였다. 특히 EdgeBox를 사용한 Fast R-CNN의 경우 mAP가 20%로 매우 낮게 나타나, RPN(Region Proposal Network)을 통한 영역 제안 방식이 수술 도구 탐지에 훨씬 효율적임을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 실제 수술 환경에서 캡처된 데이터를 바탕으로 Faster R-CNN의 효율성을 입증하여, 실시간 수술 보조 시스템의 가능성을 보여주었다는 점이다. 특히 기존의 EdgeBox나 Selective Search 같은 외부 영역 제안 방식보다 네트워크 내부에서 학습되는 RPN이 수술 도구의 특성을 더 잘 포착한다는 것을 정량적으로 보여주었다.

하지만 몇 가지 한계점과 논의 사항이 존재한다. 첫째, Fast R-CNN과의 mAP 차이가 약 3% 정도로 비교적 작다. 비록 일관되게 더 높은 성능을 보였으나, 괄목할 만한 성능 향상을 위해서는 단순한 아키텍처 적용 이상의 전략이 필요해 보인다. 둘째, 수술 도구의 도메인 특성에 특화된 데이터로 사전 학습을 진행하고 그 가중치를 전이 학습(transfer learning) 시킨다면 더욱 정교한 탐지가 가능할 것이다.

마지막으로, 본 논문에서는 YOLO나 SSD와 같은 Single-stage detector를 언급하며, 이러한 모델을 도입할 경우 탐지 속도를 더욱 높이고 성능을 개선할 수 있을 것이라는 가능성을 제시하였다. 이는 향후 연구에서 다중 모달(multimodal) 접근 방식과 결합하여 수술 단계 인식 시스템으로 확장될 수 있는 중요한 지점이 된다.

## 📌 TL;DR

본 논문은 TKA(인공 무릎 관절 치환술) 수술 중 사용되는 다양한 도구를 실시간으로 인식하기 위해 **Faster R-CNN(VGG-16 backbone)** 기반의 시스템을 제안하였다. 실제 수술 영상을 활용한 실험 결과, **mAP 87.6%**와 **탐지 시간 0.075초**를 달성하여 기존의 Fast R-CNN 및 DPM 방식보다 우수한 성능을 보였다. 이 연구는 스마트 글래스를 통한 실시간 수술 보조 및 향후 수술 워크플로우 분석을 위한 핵심 기초 기술로서 가치가 높다.
