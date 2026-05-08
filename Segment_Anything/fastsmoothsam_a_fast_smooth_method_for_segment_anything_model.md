# FastSmoothSAM: A Fast Smooth Method For Segment Anything Model

Jiasheng Xu, Yewang Chen (2025)

## 🧩 Problem to Solve

본 논문은 이미지 세그멘테이션 분야의 혁신적인 모델인 Segment Anything Model (SAM)과 그 경량화 버전인 FastSAM이 가진 한계점을 해결하고자 한다. SAM은 뛰어난 제로샷(zero-shot) 세그멘테이션 능력을 보여주지만, Vision Transformer (ViT) 백본으로 인해 과도한 GPU 메모리 소비와 느린 추론 속도라는 치명적인 단점이 있어 실시간 응용 분야에 적용하기 어렵다.

이를 해결하기 위해 제안된 FastSAM은 YOLOv8-seg를 기반으로 하여 처리 속도를 획기적으로 높였으나, 또 다른 문제인 **계단 현상(jagged edges)**이 발생한다는 점이 지적된다. 특히 정밀한 경계 정보가 필수적인 의료 영상 분석이나 고정밀 지도 제작과 같은 분야에서는 이러한 거친 경계선이 후속 분석의 정확도를 떨어뜨리는 요인이 된다. 따라서 본 연구의 목표는 FastSAM의 실시간 처리 능력을 유지하면서도, B-Spline 곡선 피팅(curve fitting) 기술을 통해 객체의 경계선을 매끄럽고 정확하게 정밀화(refinement)하는 FastSmoothSAM 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN 기반 네트워크가 생성한 거친 마스크 경계선을 수학적인 B-Spline 곡선으로 근사화하여 시각적 품질과 분석적 정확도를 높이는 것이다. 주요 기여 사항은 다음과 같다.

1. **B-Spline 곡선 피팅의 도입**: CNN 네트워크가 식별한 객체의 경계선을 매끄럽게 만들기 위해 B-Spline 곡선 피팅 기술을 적용하는 후처리 방식을 제안한다.
2. **2단계 평활화 전략 (Two-Round-Smoothing Strategy)**: 1차적으로 객체의 대략적인 윤곽선을 추정하는 Coarse fitting을 수행하고, 2차적으로 정밀한 경계 검출을 수행하는 Fine fitting을 적용하여 단계적으로 정밀도를 높인다.
3. **4단계 경계 곡선 피팅 방법 (Four-Stage-Edge-Curve-Fitting Method)**: 거친 윤곽선 추출 $\rightarrow$ Canny 에지 검출 및 영역 확장 $\rightarrow$ 적응형 샘플링(Adaptive Sampling) $\rightarrow$ 정밀 곡선 피팅으로 이어지는 파이프라인을 구축하여 경계 세그멘테이션의 품질을 극대화한다.

## 📎 Related Works

### Segment Anything Model (SAM)

SAM은 프롬프트(텍스트, 좌표, 바운딩 박스 등)에 따라 유연하게 동작하는 기초 모델로, 제로샷 성능이 매우 뛰어나다. 그러나 ViT 기반의 구조로 인해 하드웨어 요구 사양이 매우 높다는 한계가 있다.

### Fast Segment Anything (FastSAM)

SAM의 실시간 요구사항을 충족하기 위해 YOLOv8-seg를 백본으로 사용하는 CNN 기반의 인스턴스 세그멘테이션 방식을 채택하였다. 이를 통해 SAM보다 최대 50배 빠른 속도를 달성했으나, 하향 샘플링(downsampling)과 특징 추출의 한계로 인해 결과물의 경계선이 매끄럽지 못한 jagged edge 문제가 발생한다.

### B-Spline Curve

B-Spline 곡선은 제어점(control points)과 기저 함수(basis functions)의 선형 조합으로 생성되는 곡선으로, 형태 제어 능력이 뛰어나고 유연하다. 특히 최소자승법(least squares method)을 사용할 경우 계산 속도가 매우 빠르며, 이미지 처리 분야에서 객체의 윤곽선을 모델링하거나 안티앨리어싱(anti-aliasing)을 구현하는 데 적합하다.

## 🛠️ Methodology

### 전체 시스템 구조

FastSmoothSAM의 전체 파이프라인은 YOLOv8-seg를 통한 초기 세그멘테이션 이후, 네 가지 단계의 후처리 과정을 거쳐 최종 마스크를 생성한다.

### B-Spline 곡선의 수학적 정의

B-Spline 곡선 $C(t)$는 $n+1$개의 제어점 $P = [P_0, P_1, \dots, P_n]^T$와 매듭 벡터(knot vector) $U$에 의해 다음과 같이 정의된다.

$$C(t) = \sum_{i=0}^{n} N_{i,k}(t)P_i$$

여기서 $N_{i,k}(t)$는 Cox-de Boor 재귀 공식에 의해 생성되는 B-spline 기저 함수이다. 곡선의 곡률(curvature) $K$는 다음과 같이 계산된다.

$$K = \frac{\|C'(t) \times C''(t)\|}{\|C'(t)\|^3}$$

### Approximate B-Spline Fitting (최소자승법)

주어진 데이터 포인트 세트 $D$에 대해 곡선이 이를 최대한 가깝게 지나도록 하는 제어점 $P$를 찾기 위해 최소자승법을 사용한다. 데이터 포인트와 기저 함수 행렬 $N$의 관계를 $D = NP$로 정의할 때, 최적의 제어점 $P$는 다음과 같이 도출된다.

$$P = (N^T N)^{-1} N^T D$$

### 4단계 정밀화 프로세스 (Four-Stage-Edge-Curve-Fitting)

**1단계: Coarse Fitting (거친 피팅)**
YOLOv8-seg가 생성한 초기 윤곽선에서 정렬된 에지 포인트들을 추출하고, 3차 B-Spline 곡선 피팅을 수행한다. 이 단계의 결과물은 전체적인 형태는 잡지만 고곡률 영역에서 과하게 매끄러워지는(over-smoothing) 경향이 있다.

**2단계: Dilated Edge Canny Detection (확장 경계 캐니 검출)**
1단계의 결과 곡선을 확장(dilate)하여 마스크 영역 $\text{REG}_{\text{mask}}$를 생성한다. 이 영역 내에서 원본 이미지에 Canny 알고리즘을 적용하여 정밀한 에지 후보군 $\text{CAD}_{\text{canny}}$를 추출한다. 이때 가우시안 필터의 $\sigma$ 값을 조정하여 미세한 색상 차이로 인한 검출 오류를 줄인다.

**3단계: Adaptive Sampling (적응형 샘플링)**
B-Spline 피팅 시 고곡률 영역의 정보 손실을 막기 위해 샘플링 밀도를 조절한다.

- **Curvature-based sampling**: 1단계 곡선의 곡률 $K$를 계산하여 $\theta$ 임계값보다 큰 고곡률 지점을 집중적으로 샘플링한다.
- **Canny-based sampling**: 균일 샘플 $P_{\text{unif}}$를 기준으로 세 가지 케이스에 따라 포인트를 재배치한다.
  - Case 1 (Canny 에지 누락 시): 원래의 샘플 포인트 $p_i$를 그대로 사용한다.
  - Case 2 (고곡률 영역): $\text{CAD}_{\text{canny}}$의 지역 중심점을 선택하여 정밀도를 높인다.
  - Case 3 (기타 영역): $\text{CAD}_{\text{canny}}$와 $P_{\text{unif}}$의 중심점을 계산하여 샘플 밀도를 낮춘다.

**4단계: Fine Fitting (정밀 피팅)**
적응형 샘플링을 통해 얻은 정렬된 핵심 샘플 $P_s$를 사용하여 2차 B-Spline 곡선 피팅을 최종적으로 수행한다.

## 📊 Results

### 실험 환경 및 데이터셋

- **환경**: Intel i7-12700, NVIDIA GTX 3060Ti 및 RTX 3090.
- **데이터셋**: COCO, BSDS500, PASCAL VOC.
- **비교 대상**: SAM, FastSAM, MobileSAM, UnSAM, U-Net.

### 정량적 결과

1. **실행 시간 및 메모리**:
    - FastSmoothSAM은 FastSAM보다 마스크당 약 0.7~2ms 정도 느리지만, SAM이나 MobileSAM보다는 압도적으로 빠르다.
    - 메모리 소비량은 SAM이 가장 높으며, FastSmoothSAM은 FastSAM과 유사하거나 약간 낮은 수준을 유지하여 효율성을 입증하였다.
2. **평가 지표 (Curvature, Fréchet Distance)**:
    - 평균 곡률(Mean Curvature)과 곡률 분산(Curvature Variance) 측정 결과, FastSmoothSAM이 FastSAM 및 U-Net보다 더 매끄러운 경계선을 생성함을 확인하였다.
    - 두 곡선 간의 유사도를 측정하는 Fréchet Distance (FD)에서 FastSmoothSAM이 가장 낮은 값을 기록하여 Ground Truth에 가장 근접한 형태임을 보였다.

### 정성적 결과 및 한계

- **시각적 개선**: 펜타그램 및 다양한 COCO 데이터셋 실험에서 FastSAM의 계단 현상이 제거되고 매끄러운 곡선 형태가 복원됨을 확인하였다.
- **한계점**: B-Spline의 수학적 특성상 **날카로운 모서리(sharp edges)**를 가진 객체를 피팅할 때 모서리가 뭉툭해지는 현상이 발생하며, **매우 작은 객체**의 경우 샘플 포인트 부족으로 인해 형태가 왜곡되는 문제가 발견되었다.

## 🧠 Insights & Discussion

본 논문은 딥러닝 모델의 아키텍처를 직접 수정하여 경계선을 개선하려는 시도(예: Edge-aware loss 추가 등)가 비용이 많이 든다는 점에 착안하여, 효율적인 수학적 후처리 방식을 제안하였다. 특히 단순히 곡선을 피팅하는 것에 그치지 않고, Canny 에지 검출과 곡률 기반의 적응형 샘플링을 결합하여 "데이터 기반의 정밀도"와 "수학적 매끄러움" 사이의 균형을 맞춘 점이 돋보인다.

다만, B-Spline이라는 도구의 본질적인 특성이 '매끄러움'에 치중되어 있어, 모든 객체에 만능은 아니라는 점이 명확히 드러났다. 직사각형이나 삼각형처럼 각진 형태의 객체에 대해서는 오히려 과도한 평활화가 독이 될 수 있다. 향후 연구에서는 객체의 기하학적 특성(각진 정도)을 먼저 판단하고 피팅 전략을 다르게 가져가는 적응형 알고리즘이 필요할 것으로 보인다.

## 📌 TL;DR

FastSmoothSAM은 FastSAM의 빠른 속도는 유지하면서, 고질적인 문제인 경계선 계단 현상을 해결하기 위해 **B-Spline 곡선 피팅 기반의 4단계 정밀화 공정**을 도입한 모델이다. 적응형 샘플링을 통해 고곡률 영역의 디테일을 보존하면서도 매끄러운 경계를 생성하며, 이는 실시간성이 요구되는 정밀 의료 영상 및 산업 자동화 분야에서 매우 유용할 것으로 기대된다. 다만, 매우 작은 객체나 날카로운 모서리를 가진 객체에 대해서는 피팅 정확도가 떨어진다는 한계가 존재한다.
