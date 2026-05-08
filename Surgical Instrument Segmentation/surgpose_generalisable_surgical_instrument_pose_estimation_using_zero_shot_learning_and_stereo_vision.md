# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision

Utsav Rai, Haozheng Xu, and Stamatia Giannarou (2025)

## 🧩 Problem to Solve

로봇 보조 최소 침습 수술(Robot-assisted Minimally Invasive Surgery, RMIS)에서 수술 도구의 정확한 포즈 추정(Pose Estimation)은 수술 내비게이션과 로봇 제어를 위해 필수적이다. 하지만 기존의 접근 방식들은 다음과 같은 심각한 한계를 가지고 있다.

첫째, 마커 기반(Marker-based) 방법은 폐색(Occlusion), 반사, 도구별 맞춤 설계의 필요성 등의 문제가 있으며, 특히 케이블 구동 로봇 시스템의 경우 기구학적(Kinematic) 측정값의 캘리브레이션 부족으로 인해 부정확한 결과가 발생한다.

둘째, 지도 학습(Supervised Learning) 기반 방법은 마커 없이 가능하지만, 새로운 도구를 도입할 때마다 대규모의 어노테이션된 데이터셋이 필요하므로 일반화 능력이 부족하다.

셋째, 기존의 Zero-shot 포즈 추정 모델들은 주로 RGB-D 센서에 의존한다. 그러나 수술 환경은 근접 촬영 환경이며, 금속 도구의 강한 반사와 센서의 블라인드 존(Blind zone) 문제로 인해 일반적인 깊이 센서를 사용하기 어렵다.

따라서 본 논문의 목표는 수술 환경의 특수성(반사, 폐색, 깊이 센서 사용 불가)을 극복하면서도, 학습되지 않은 새로운 도구에 대해 일반화 가능한 Zero-shot 6DoF(6 Degrees of Freedom) 포즈 추정 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 Zero-shot RGB-D 포즈 추정 프레임워크에 수술 환경에 최적화된 **스테레오 기반 깊이 추정(Stereo-based Depth Estimation)**과 **정밀한 인스턴스 분할(Instance Segmentation)** 모듈을 결합하는 것이다.

주요 기여 사항은 다음과 같다.

1. 하드웨어 깊이 센서 대신 **RAFT-Stereo** 알고리즘을 도입하여, 반사가 심하고 텍스처가 없는 수술 환경에서도 강건한 Pseudo-depth를 생성함으로써 RGB-D 모델의 적용 가능성을 확장하였다.
2. SAM-6D 모델의 기존 분할 모듈인 SAM(Segment Anything Model)을 수술 도구 데이터로 미세 조정(Fine-tuning)된 **Mask R-CNN**으로 교체하여, 폐색 및 복잡한 환경에서의 분할 정확도를 크게 향상시켰다.
3. 실제 및 합성 스테레오 이미지와 Ground Truth 포즈 정보를 포함하는 새로운 수술 도구 포즈 추정 데이터베이스를 구축하였다.
4. 다양한 Zero-shot 모델(FoundationPose, SAM-6D, OVE-6D, MegaPose)을 RMIS 환경에서 종합적으로 평가하여 제안 방법의 우수성을 입증하였다.

## 📎 Related Works

### 1. RGB 기반 접근 방식

- **간접 방법(Indirect Methods):** 2D 키포인트를 검출하고 3D 모델과 매칭하여 PnP(Perspective-n-Point) 알고리즘으로 포즈를 추정한다. 폐색에 강건하지만, 반복적인 비미분 가능 솔버를 사용하여 엔드투엔드 학습이나 실시간 적응성이 떨어진다.
- **직접 방법(Direct Methods):** 입력 이미지에서 6D 포즈를 직접 회귀(Regression)한다. 미분 가능한 프로세스로 구현 가능하지만, 대규모 어노테이션 데이터셋이 필요하여 새로운 도구에 대한 일반화가 어렵다.

### 2. RGB-D 기반 접근 방식

깊이 데이터를 함께 사용하여 복잡한 환경에서 정확도를 높인다. 하지만 RMIS에서는 센서의 물리적 한계로 인해 정확한 깊이 데이터를 얻기 어렵다는 점이 가장 큰 제약이다.

### 3. Zero-shot 포즈 추정

FoundationPose, SAM-6D, OVE-6D 등이 있으며, 특정 객체에 대한 추가 학습 없이 일반화가 가능하다. 본 논문은 이러한 모델들이 필요로 하는 '깊이 정보'를 센서가 아닌 스테레오 비전으로 해결하고, '분할 성능'을 Mask R-CNN으로 보완함으로써 RMIS에 적용하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

제안하는 파이프라인은 다음과 같은 순서로 진행된다:
$\text{Stereo Images} \rightarrow \text{RAFT-Stereo (Depth Estimation)} \rightarrow \text{Mask R-CNN (Tool Segmentation)} \rightarrow \text{Zero-shot Pose Estimation Model}$

### 2. 스테레오 기반 깊이 추정 (Stereo-Based Depth Estimation)

- **RAFT-Stereo:** Rectified 스테레오 이미지 쌍으로부터 시차(Disparity)를 계산한다. 3D 상관 볼륨(Correlation volume)을 구축하고 다층 GRU(Gated Recurrent Units)를 통해 시차 추정치를 반복적으로 정밀화한다. 이는 금속 도구의 반사나 텍스처가 없는 영역에서도 강건한 성능을 보인다.
- **깊이 계산:** 계산된 시차로부터 각 픽셀의 깊이 $Z(x,y)$는 스테레오 베이스라인 $B$와 카메라 초점 거리 $f$를 이용하여 다음과 같이 계산된다. (논문 내 명시적 수식은 없으나 일반적인 스테레오 기하학을 따름)

### 3. Mask R-CNN을 이용한 마스크 생성

Zero-shot 분할 모델(SAM 등)이 수술 환경의 폐색 및 반사 상황에서 성능이 저하되는 문제를 해결하기 위해 미세 조정된 Mask R-CNN을 사용한다.

- **Pseudo Label 생성:** 3D CAD 모델을 2D 이미지 평면으로 투영하여 초기 마스크를 생성한 후, 투영된 모델의 깊이 $Z_{proj}(u,v)$와 RAFT-Stereo로 얻은 깊이 $Z_{disp}(u,v)$를 비교하여 실제 가시 영역만 남긴다.
$$|Z_{proj}(u,v) - Z_{disp}(u,v)| < \epsilon$$
여기서 $\epsilon$은 $1\text{mm}$로 설정되며, 이를 통해 폐색된 영역을 제거하고 가시적인 단면만 보존한다.
- **학습 전략:** NVISII 렌더러를 이용한 광학적 합성 데이터(Synthetic data)와 실제 이미지(Real images)를 혼합하여 학습시킴으로써 데이터 수집 비용을 줄이고 일반화 성능을 높였다.

### 4. Zero-Shot RGB-D 포즈 추정

최종 단계에서는 RGB 이미지, RAFT-Stereo로 생성한 깊이 맵, 그리고 Mask R-CNN으로 생성한 마스크를 입력으로 하여 포즈를 추정한다.

- 특히 **SAM-6D** 프레임워크에서 기존의 SAM 모듈을 제안한 **Mask R-CNN**으로 교체하여 적용하였다.
- 그 외 FoundationPose, OVE-6D, MegaPose 모델들에도 동일하게 Mask R-CNN 기반의 마스크를 입력으로 제공하여 성능을 비교하였다.

## 📊 Results

### 1. 실험 설정

- **대상 도구:** Da Vinci™ Si 시스템의 Endowrist™ Large Needle Driver (LND).
- **데이터셋:**
  - **Dataset A:** 폐색이 없는 상태의 LND 이미지 (1,027장).
  - **Dataset B:** 다른 수술 도구에 의해 LND가 부분적으로 폐색된 이미지 (797장).
  - **Dataset C:** Mask R-CNN 학습용 데이터 (실제 1,489장 + 합성 1,000장).
- **평가 지표:**
  - **분할:** $\text{AP}@[IoU=0.50:0.95]$, $\text{AR}@[IoU=0.50:0.95]$.
  - **포즈:** $\text{ADD}$ (Average Distance of Model Points), 2D Projection 거리.

### 2. 분할 성능 결과

Mask R-CNN (ResNet-101 backbone)은 AP $86.9$를 기록하며, SAM (AP $46.4$)보다 압도적으로 높은 성능을 보였다. 이는 수술 환경의 특수한 폐색 상황에서 지도 학습 기반의 미세 조정이 훨씬 효과적임을 시사한다.

### 3. 포즈 추정 결과

- **비폐색 상황 (Non-Occluded):**
  - $\text{ADD (5mm threshold)}$ 기준, FoundationPose가 $48.29\%$로 가장 높았으나, 제안하는 $\text{SAM-6D (Mask R-CNN)}$ 또한 $46.86\%$로 매우 근접한 성능을 보였다. (원본 SAM-6D는 $0.88\%$에 불과함)
  - $\text{2D Projection (50px)}$ 기준으로는 $\text{SAM-6D (Mask R-CNN)}$가 $95.14\%$로 FoundationPose($90.29\%$)를 능가하였다.
- **폐색 상황 (Occluded):**
  - $\text{SAM-6D (Mask R-CNN)}$가 $\text{ADD (5mm)}$ 기준 $49.06\%$를 기록하며 FoundationPose ($6.02\%$)를 압도하였다.
  - $\text{2D Projection (50px)}$ 역시 $98.87\%$의 높은 정확도를 보였다.
- **통계적 일관성:** $\text{SAM-6D (Mask R-CNN)}$의 $\text{ADD}$ 평균값($\mu$)은 $6.47\text{mm}$로 FoundationPose($8.05\text{mm}$)보다 정밀하며, 표준편차($\sigma$) 또한 $9.56\text{mm}$로 훨씬 낮아 결과의 일관성이 높음을 확인하였다.

### 4. 실행 속도

첫 프레임 초기화에는 $2.52\text{s}$가 소요되지만, 이후 프레임부터는 약 $15\text{fps}$의 속도로 거의 실시간 추적이 가능하다.

## 🧠 Insights & Discussion

본 연구는 Zero-shot 포즈 추정 모델이 실제 수술 환경에서 작동하기 위해 가장 필요한 두 가지 요소가 **'신뢰할 수 있는 깊이 정보'**와 **'정교한 객체 마스크'**임을 입증하였다.

특히 흥미로운 점은, 일반적인 Zero-shot 모델(SAM-6D 등)의 성능 저하 원인이 포즈 추정 알고리즘 자체보다는 입력값인 마스크의 품질에 있다는 것이다. SAM과 같은 범용 모델보다 수술 도구에 특화되어 학습된 Mask R-CNN을 사용했을 때 포즈 정확도가 비약적으로 상승한 점이 이를 뒷받침한다.

또한, 하드웨어 깊이 센서를 사용할 수 없는 RMIS 환경에서 RAFT-Stereo를 통한 Pseudo-depth 생성 방식이 충분히 유효한 대안이 될 수 있음을 보여주었다.

다만, 여전히 지도 학습 기반의 방법(예: PVNet)보다는 절대적인 정확도가 낮을 수 있다. 하지만 새로운 도구가 도입될 때마다 데이터를 수집하고 학습시켜야 하는 비용을 고려하면, 제안된 Zero-shot 파이프라인의 범용성은 실제 임상 적용에서 훨씬 더 큰 가치를 가진다.

## 📌 TL;DR

이 논문은 수술 로봇 환경에서 센서 없이 스테레오 비전(RAFT-Stereo)과 미세 조정된 분할 모델(Mask R-CNN)을 결합하여, 학습되지 않은 새로운 수술 도구의 포즈를 추정하는 **SurgPose** 파이프라인을 제안한다. 실험 결과, 특히 도구가 다른 물체에 가려지는 폐색 상황에서 기존의 최신 Zero-shot 모델들보다 훨씬 강건하고 정확한 6DoF 포즈 추정 성능을 보였으며, 이는 향후 다양한 수술 도구에 즉각적으로 적용 가능한 범용적 수술 내비게이션 시스템 구축에 기여할 것으로 기대된다.
