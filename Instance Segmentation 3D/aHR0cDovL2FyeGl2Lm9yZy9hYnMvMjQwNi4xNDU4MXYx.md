# 3D Instance Segmentation Using Deep Learning on RGB-D Indoor Data

Siddiqui Muhammad Yasir, Amin Muhammad Sadiq and Hyunsik Ahn (2022)

## 🧩 Problem to Solve

본 논문은 산업 및 가정 내 실내 환경에서 지능형 로봇 시스템이 마주하는 객체들을 인식하고 분리하는 3D Instance Segmentation 문제를 해결하고자 한다. 3D 객체 인식은 로봇의 객체 핸들링, 자율 주행 차량의 상황 인지, 공장 자동화 비전 시스템 등 다양한 분야에서 필수적이다.

기존의 3D 세그멘테이션 방식은 수작업으로 설계된 feature(hand-crafted features)에 의존하여 성능이 낮고 대규모 데이터셋으로의 일반화가 어려웠다. 최근에는 딥러닝 기반의 접근 방식이 도입되었으나, Point Cloud 데이터를 직접 처리하는 방식은 데이터의 불규칙성으로 인해 RGB-D 채널의 특징을 병합하기 어렵고, 고해상도 Voxel로 변환할 때 막대한 계산 비용과 자원이 소모되는 문제가 있다. 따라서 본 연구의 목표는 전처리를 간소화하고 자원 소모를 줄이면서도, 정밀한 3D 기하학적 정보를 제공할 수 있는 효율적인 3D Instance Segmentation 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 2D Instance Segmentation의 높은 성능과 Depth 데이터를 결합하여, 계산 효율성을 높이면서도 정확한 3D 좌표를 추출하는 파이프라인을 구축하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **2D-to-3D 통합 프레임워크 제안**: 딥러닝 기반의 2D 객체 인식 결과와 Depth 정보를 통합하여 3D Instance Segmentation을 수행하는 새로운 방법을 제시하였다.
2.  **정밀한 경계 추출을 위한 PointRend 적용**: Mask R-CNN의 기본 마스크 헤드가 가진 '뭉툭한(blobby)' 출력 문제를 해결하기 위해 Point-based Rendering(PointRend) 모듈을 도입하여 객체의 경계를 정밀하게 추출하였다.
3.  **객체와 배경의 명확한 분리**: 단순히 객체를 세그멘테이션하는 것에 그치지 않고, foreground 객체와 background 정보를 개별적인 3D 데이터로 분리하여 추출함으로써 로봇 시스템의 환경 인지 능력을 높였다.

## 📎 Related Works

논문에서는 2D 및 3D 객체 인식에 관한 기존 연구들을 다음과 같이 설명한다.

-   **2D Instance Recognition**: CNN 기반의 YOLO, Fast/Faster R-CNN 등이 언급된다. 특히 Mask R-CNN은 객체의 클래스, 바운딩 박스, 마스크를 동시에 예측하는 state-of-the-art 모델로 평가받는다. 다만, 큰 객체의 경우 경계가 과도하게 부드러워지는 경향이 있는데, 이를 해결하기 위해 적응적 샘플링을 통해 정밀한 경계를 생성하는 PointRend 기술이 제안되었다.
-   **3D Object Recognition**: PointNet과 같이 Point Cloud에서 직접 특징을 학습하는 방식이 있으나, 이는 계산 비용이 매우 높고 전처리가 복잡하다는 한계가 있다. 또한, 3D Bounding Box를 예측하는 방식은 실제 로봇의 grasping(잡기) 시나리오에서는 세밀한 기하학적 정보가 부족하여 부적합하다.
-   **차별점**: 제안 방법은 Point Cloud를 직접 입력으로 사용하지 않고, RGB 이미지에서 2D 세그멘테이션을 먼저 수행한 후 이를 Depth 맵에 투영하는 방식을 취함으로써 연산 비용을 획기적으로 줄이고 실시간 로봇 응용 가능성을 높였다.

## 🛠️ Methodology

본 논문이 제안하는 3D Instance Segmentation 파이프라인은 크게 세 단계로 구성된다.

### 1. 2D Masking 및 인식
먼저 RGB-D 데이터 중 RGB 이미지($S^R$)를 입력으로 하여 **Mask R-CNN**과 **PointRend** 모듈을 통해 타겟 객체를 인식하고 마스킹한다. 
-   Mask R-CNN이 객체의 대략적인 영역(ROI)과 클래스를 예측하면, PointRend가 적응적 샘플링을 통해 경계를 정밀하게 다듬어 Boolean 형태의 이진 마스크(0 또는 1)를 생성한다.

### 2. ROI 분리 및 세그멘테이션
생성된 이진 마스크를 바탕으로 RGB 및 Depth 이미지에서 관심 영역(ROI)을 분리한다.
-   **Contour Detection**: 마스크의 윤곽선(contour)을 검출하여 객체의 영역을 확정한다.
-   **Depth Thresholding**: 센서에서 발생하는 노이즈나 불필요한 픽셀을 제거하기 위해, 센서와 객체 사이의 거리 범위를 설정하는 임계값(threshold) 필터링을 적용한다.

### 3. 3D 좌표 투영 (Perspective Transformation)
분리된 2D 픽셀 좌표 $(u, v)$와 해당 위치의 Depth 값 $d$를 이용하여 3D 공간 좌표 $(x, y, z)$로 변환한다. 이때 카메라 캘리브레이션을 통해 얻은 내적 행렬(Intrinsic Matrix)의 초점 거리 $f_x, f_y$를 사용하며, 변환 식은 다음과 같다.

$$z = d$$
$$x = \frac{u \cdot z}{f_x}$$
$$y = \frac{v \cdot z}{f_y}$$

(논문 내 식 (1)에서는 $z$를 구할 때 분모에 $\sqrt{1+(u/f_x)^2 + (v/f_y)^2}$ 형태의 보정이 언급되어 있으나, 일반적인 핀홀 카메라 모델의 투영 방식을 기반으로 한다.)

최종적으로 이렇게 계산된 3D 포인트들을 Open3D 라이브러리를 통해 시각화하여 3D Instance Point Cloud를 생성한다.

## 📊 Results

### 실험 환경 및 설정
-   **센서**: Intel RealSense D-415 (RGB-D 카메라)
-   **하드웨어**: NVIDIA GeForce 1080Ti GPU, Intel Core i7 CPU
-   **데이터셋**: MS COCO 데이터셋으로 사전 학습된 Mask R-CNN 모델 사용
-   **측정 거리**: 객체로부터 3~4m 거리에서 촬영

### 정량적 결과
1.  **모델별 인식률 비교**: 동일 데이터셋에 대해 Mask R-CNN이 가장 높은 인식률을 보였다.
    -   Single Object: Mask R-CNN (98%) > Faster R-CNN (94%) > Fast R-CNN (90%)
    -   Multi Objects: Mask R-CNN (92%) > Faster R-CNN (89%) > Fast R-CNN (84%)
2.  **실제 치수 대비 오차 분석**: 세그멘테이션 된 3D 객체의 치수를 실제 물리적 치수와 비교한 결과, 매우 낮은 오차를 보였다.
    -   책(너비): 실제 203mm $\rightarrow$ 측정 200mm (오차 3mm)
    -   컵(너비): 실제 76mm $\rightarrow$ 측정 74mm (오차 2mm)
    -   시계(지름): 실제 228mm $\rightarrow$ 측정 227mm (오차 1mm)

### 정성적 결과
실험 결과, 배경과 타겟 객체를 명확히 분리하여 각각의 3D 포인트 클라우드로 저장할 수 있음을 확인하였다. 특히 단일 객체뿐만 아니라 여러 객체가 섞여 있는 환경에서도 각 인스턴스를 개별적으로 분리해내는 성능을 보였다.

## 🧠 Insights & Discussion

### 강점
본 연구의 가장 큰 강점은 **배경 분리(Background Separation)** 능력이 있다는 점이다. 기존 연구들이 객체 세그멘테이션에만 집중한 반면, 본 논문은 객체를 제거한 배경 정보와 객체 정보를 따로 추출함으로써 로봇이 주변 환경과 타겟 객체를 동시에 인지해야 하는 실제 상황에 매우 유용한 데이터를 제공한다. 또한, 2D 딥러닝 모델을 활용함으로써 3D 데이터를 직접 처리하는 것보다 연산 효율성을 크게 높였다.

### 한계 및 미해결 과제
-   **폐색(Occlusion) 문제**: 객체가 서로 겹쳐 있거나 가려진 경우, 2D 마스크 생성 단계에서 오류가 발생하여 3D 세그멘테이션 결과가 혼합되는 문제가 발생한다.
-   **센서 노이즈**: 카메라 센서 특성 및 환경적 요인으로 인해 3D 뷰에서 일부 노이즈가 관찰된다.

### 비판적 해석
제안된 방법론은 매우 직관적이며 효율적이지만, 본질적으로 2D 세그멘테이션의 성능에 전적으로 의존한다. 즉, Mask R-CNN이 객체를 잘못 인식하면 3D 좌표 역시 잘못 생성될 수밖에 없는 구조이다. 하지만 로봇의 grasping 작업에서 1~3mm 정도의 오차는 무시 가능한 수준이라는 저자의 주장은 실용적인 관점에서 타당해 보이며, 복잡한 3D 네트워크 없이도 충분한 정밀도를 얻을 수 있음을 입증하였다.

## 📌 TL;DR

본 논문은 **Mask R-CNN과 PointRend를 이용한 2D 인스턴스 세그멘테이션 결과에 Depth 정보를 결합하여, 저비용·고효율로 3D 객체를 분리하는 방법**을 제안한다. 특히 객체뿐만 아니라 배경을 별도의 3D 포인트 클라우드로 분리해낼 수 있다는 점이 특징이며, 실제 측정 오차가 1~3mm 수준으로 매우 적어 로봇의 객체 조작(grasping) 및 환경 인지 시스템에 즉각적으로 적용 가능한 실용적인 연구이다.