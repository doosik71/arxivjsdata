# Deep Learning Based Instance Segmentation in 3D Biomedical Images Using Weak Annotation

Zhuo Zhao, Lin Yang, Hao Zheng, Ian H. Guldner, Siyuan Zhang, and Danny Z. Chen (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 3D 생의학 이미지(Biomedical Images)에서 인스턴스 분할(Instance Segmentation)을 수행하기 위해 필요한 막대한 양의 어노테이션 비용 문제이다. 일반적으로 3D 인스턴스 분할을 위한 딥러닝 모델을 학습시키려면 모든 인스턴스의 모든 복셀(voxel)을 정밀하게 마스킹하는 Full Voxel Annotation이 필요하다. 그러나 이러한 작업은 고도의 전문 지식을 갖춘 전문가만이 수행할 수 있으며, 3D 데이터의 특성상 작업량이 매우 많아 시간과 비용이 과도하게 소모된다.

또한, 기존의 2D 약지도 학습(Weakly Supervised Learning) 방식들은 3D 이미지에 직접 적용하기 어렵고, 2D 분할 결과를 단순히 쌓아(stacking) 3D로 확장하는 방식은 3D 문맥 정보(Context Information)를 활용하지 못해 정확도가 낮으며 연산 복잡도가 높아 처리 시간이 매우 오래 걸린다는 한계가 있다. 따라서 본 연구의 목표는 전체 복셀 어노테이션 없이, 상대적으로 적은 비용이 드는 약한 어노테이션(Weak Annotation)만을 활용하여 빠르고 효율적인 3D 인스턴스 분할 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 모든 인스턴스에 대한 3D Bounding Box 어노테이션과 아주 적은 비율의 인스턴스에 대해서만 제공되는 Full Voxel Annotation을 결합하여 모델을 학습시키는 약지도 학습 접근법이다. 이를 위해 모델을 두 단계(Two-stage)로 설계하여, 첫 번째 단계에서는 Bounding Box를 이용해 모든 인스턴스를 검출하고, 두 번째 단계에서는 일부 마스크 데이터를 이용해 검출된 인스턴스의 세부 영역을 분할하도록 한다. 이러한 설계를 통해 어노테이션 시간을 획기적으로 줄이면서도 전체 복셀 어노테이션을 사용한 모델과 유사한 성능을 달성하고자 하였다.

## 📎 Related Works

논문에서는 2D 영역의 인스턴스 분할 및 검출 방법론인 Faster-RCNN과 Mask R-CNN을 언급한다. 특히 Mask R-CNN은 RPN(Region Proposal Network)을 통해 관심 영역(RoI)을 제안하고, RoIAlign을 통해 정교한 마스크를 생성하는 구조를 가지고 있다. 

기존의 3D 접근 방식 중 하나는 2D 픽셀 분할 결과를 3D 복셀 분할 결과로 쌓아 올린 뒤, 고복잡도 알고리즘을 적용하여 인스턴스를 분리하는 방식이다. 하지만 이 방식은 다음과 같은 한계가 있다:
1. 3D 문맥 정보를 충분히 활용하지 못해 분할 정확도가 떨어진다.
2. 알고리즘의 복잡도가 높아 밀집된 3D 데이터를 처리하는 데 수 시간이 소요된다.
3. GPU 메모리 제한으로 인해 2D 모델을 단순히 3D로 확장하는 것에 어려움이 있다.

본 논문은 이러한 한계를 극복하기 위해 3D RoIAlign과 약지도 학습 전략을 도입한 엔드투엔드(End-to-End) 3D 모델을 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
제안된 모델은 크게 **3D Object Detector**와 **3D Voxel Segmentation Model**의 두 단계로 구성된다. 전체적인 백본(Backbone) 네트워크는 정보가 순방향 및 역방향으로 직접 전파될 수 있도록 설계된 VoxRes block을 채택하였다.

### 1. 3D Object Detector (Bounding Box 기반)
첫 번째 단계에서는 모든 인스턴스의 3D Bounding Box 어노테이션을 사용하여 객체의 위치와 클래스를 예측한다.
- **Anchor Boxes**: 특징 맵(Feature Map)의 각 위치에서 다양한 크기의 앵커 박스를 평가한다. 앵커 박스와 Ground-truth 박스 간의 IoU(Intersection over Union)가 0.4 이상인 경우를 Positive로 간주한다.
- **예측 항목**: 각 앵커 박스에 대해 인스턴스 클래스 점수(Score)와 박스 회귀 오프셋(Box Regression Offset)을 예측한다.
- **회귀 방정식**: 예측된 박스의 중심 좌표 $(z, x, y)$와 크기 $(d, h, w)$를 앵커 박스 $(z_a, x_a, y_a, d_a, h_a, w_a)$ 및 Ground-truth 박스 $(z^*, x^*, y^*, d^*, h^*, w^*)$와 비교하여 다음과 같이 정의한다.
  $$t_z = (z-z_a)/d_a, \quad t_x = (y-y_a)/h_a, \quad t_y = (x-x_a)/w_a$$
  $$t_d = \log(d/d_a), \quad t_h = \log(h/h_a), \quad t_w = \log(w/w_a)$$
- **손실 함수**: RPN의 멀티태스크 손실 함수를 그대로 사용하되, 클래스 분류를 위해 $c$개의 클래스에 대한 로그 손실($L_{cls}$)과 박스 위치 오차를 위한 Smooth $L_1$ 손실($L_{reg}$)을 합산한다.
  $$L_{box} = L_{cls} + L_{reg}$$

### 2. 3D Voxel Segmentation (약한 마스크 기반)
두 번째 단계에서는 검출된 인스턴스 내부의 정확한 복셀 영역을 분할한다.
- **3D RoIAlign**: 2D RoIAlign을 3D로 확장하여, 다양한 크기의 검출 영역을 고정된 크기 $s \times s \times s \times p$로 정렬한다. 이때 각 샘플링 포인트의 값은 주변 8개 이웃 포인트에 대해 **Trilinear Interpolation(삼선형 보간법)**을 적용하여 계산함으로써 더욱 매끄럽고 정확한 마스크를 생성한다.
- **Mask-Weight Layer**: 본 모델의 핵심으로, 모든 인스턴스를 분할하지만 **학습 시에는 Full Voxel Annotation이 존재하는 소수의 인스턴스에 대해서만 역전파(Back-propagation)를 수행**한다. 마스크 가중치 레이어를 통해 어노테이션이 없는 인스턴스의 손실 값은 0으로 설정하여 모델 학습에 영향을 주지 않도록 한다.
- **최종 손실 함수**:
  $$L = L_{box} + L_{mask}$$
  여기서 $L_{mask}$는 평균 이진 교차 엔트로피(Average Binary Cross-Entropy) 손실이다.

## 📊 Results

### 실험 설정
- **데이터셋**: HL60 세포 핵, Microglia 세포, C.elegans 배아 이미지.
- **비교 대상**: VoxResNet (Full Annotation 사용 버전 및 일부 데이터 사용 버전).
- **측정 지표**: Detection F1-score 및 Segmentation F1-score.
- **어노테이션 비율**: HL60 세포의 경우 전체의 20%, Microglia 세포의 경우 30%에 대해서만 Full Voxel Mask를 사용하였다.

### 주요 결과
1. **HL60 세포 및 Microglia 세포**: 
   - **검출 성능**: 제안 방법이 Full Annotation을 사용한 VoxResNet보다 더 높은 Detection F1-score를 기록하였다. 이는 Bounding Box 어노테이션이 인스턴스 수준의 정보를 더 강하게 제공하며, 모델이 $L_{box}$를 통해 명시적으로 검출 학습을 수행하기 때문이다.
   - **분할 성능**: 정밀한 분할 성능 자체는 Full Annotation 모델이 약간 더 우세했으나, **비슷한 어노테이션 시간(AT)을 소모했을 때**는 제안 방법이 VoxResNet(일부 데이터 학습)보다 훨씬 뛰어난 성능을 보였다.
   - **효율성**: HL60 세포의 경우, 제안 방법의 어노테이션 시간은 약 5.5시간으로, Full Annotation 방식(22.5시간)보다 약 4배 이상 적은 시간이 소요되었다.

2. **C.elegans 배아**:
   - Full Voxel Annotation이 부족한 데이터셋으로, 검출 성능만을 평가하였다. Ground-truth 마커와 검출된 박스 중심 간의 거리를 기준으로 측정했을 때 F1-score 0.9495를 달성하였다.

## 🧠 Insights & Discussion

본 논문의 결과는 3D 인스턴스 분할에서 **"정확한 위치 검출"이 "정밀한 복셀 분할"의 전제 조건**이 됨을 시사한다. Bounding Box 어노테이션은 복셀 마스크보다 생성 비용이 훨씬 저렴함에도 불구하고, 모델에게 인스턴스의 경계와 위치에 대한 명확한 가이드를 제공하여 결과적으로 적은 양의 마스크 데이터만으로도 높은 분할 성능을 낼 수 있게 한다.

특히, `Mask-Weight Layer`를 통해 일부 데이터만으로 학습하면서도 전체 인스턴스를 분할할 수 있게 한 점은 실무적으로 매우 중요한 기여이다. 다만, RoIAlign 과정에서의 리샘플링 연산으로 인해 Full Annotation 모델의 최고 성능에는 약간 못 미치는 한계가 관찰되었다. 이는 3D 데이터의 해상도 손실을 최소화하면서 효율적으로 특징을 추출하는 추가적인 기법이 필요함을 의미한다.

## 📌 TL;DR

본 연구는 3D 생의학 이미지의 어노테이션 비용 문제를 해결하기 위해 **모든 인스턴스의 Bounding Box와 극소수 인스턴스의 Voxel Mask만을 사용하는 약지도 학습 기반의 2단계 3D 인스턴스 분할 모델**을 제안하였다. 3D RoIAlign과 Mask-Weight Layer를 도입하여 학습 효율을 극대화하였으며, 실험 결과 Full Annotation 방식 대비 어노테이션 시간을 획기적으로 단축하면서도 대등하거나 더 우수한 검출 및 분할 성능을 확보하였다. 이 연구는 데이터 구축 비용이 매우 높은 3D 의료 영상 분석 분야에서 실질적인 학습 전략을 제시했다는 점에서 향후 연구 및 실제 적용 가능성이 매우 높다.