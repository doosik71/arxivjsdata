# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision

Utsav Rai, Haozheng Xu and Stamatia Giannarou (2025)

## 🧩 Problem to Solve

본 논문은 로봇 보조 최소 침습 수술(Robot-assisted Minimally Invasive Surgery, RMIS) 환경에서 수술 도구의 6자유도(6 Degrees of Freedom, 6DoF) 포즈 추정(Pose Estimation) 문제를 해결하고자 한다. 정확한 포즈 추정은 수술 내비게이션과 로봇 제어의 정밀도 및 안전성을 보장하는 데 필수적이다.

기존의 접근 방식은 크게 두 가지 한계를 가진다. 첫째, 마커 기반 방식(Marker-based methods)은 반사광, 가려짐(Occlusion) 문제에 취약하며 도구마다 특수한 마커 설계가 필요하다. 둘째, 지도 학습 기반 방식(Supervised learning methods)은 도구별로 대규모의 정답 데이터셋(Annotated dataset)이 필요하므로, 새로운 도구가 도입될 때마다 재학습을 해야 하는 일반화(Generalisation)의 한계가 있다.

따라서 본 연구의 목표는 학습되지 않은 새로운 수술 도구에 대해서도 즉각적으로 포즈를 추정할 수 있는 제로샷(Zero-shot) 포즈 추정 파이프라인을 RMIS 환경에 맞게 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 제로샷 RGB-D 포즈 추정 모델들을 RMIS의 특수한 환경(반사광이 심하고 텍스처가 없는 환경)에 맞게 최적화하여 적용하는 것이다. 주요 기여 사항은 다음과 같다.

- **스테레오 비전 기반 깊이 추정 도입**: 하드웨어 깊이 센서 대신 RAFT-Stereo 알고리즘을 사용하여, 반사광이 심한 금속성 수술 도구 환경에서도 강건한 깊이 맵(Depth map)을 생성하도록 개선하였다.
- **세그멘테이션 모듈의 고도화**: SAM-6D 모델의 기본 세그멘테이션 모듈인 SAM(Segment Anything Model)을 수술 환경에 특화되어 미세 조정(Fine-tuning)된 Mask R-CNN으로 교체하여, 가려짐이나 복잡한 배경에서도 정밀한 마스크 생성이 가능하게 하였다.
- **신규 데이터베이스 구축**: 실제 수술 환경과 합성(Synthetic) 이미지를 모두 포함하며, 정답 포즈 정보가 포함된 수술 도구 포즈 추정 데이터베이스를 구축하여 벤치마크를 제공하였다.
- **제로샷 모델의 종합적 평가**: FoundationPose, SAM-6D, OVE-6D, MegaPose 등 최신 제로샷 모델들을 RMIS 환경에서 평가하고, 제안한 개선 방법의 효용성을 입증하였다.

## 📎 Related Works

### 1. RGB 기반 접근 방식

- **간접 방식(Indirect methods)**: 2D-3D 대응 관계를 설정하고 PnP(Perspective-n-Point) 알고리즘을 사용하여 포즈를 추정한다. 가려짐에 강건하지만, 반복적인 비미분 가능 솔버(Non-differentiable solvers)를 사용하므로 실시간 적응성이나 end-to-end 학습에 제약이 있다.
- **직접 방식(Direct methods)**: 입력 이미지에서 6D 포즈를 직접 회귀(Regression)한다. 미분 가능한 프로세스로 구현 가능하지만, 대규모의 정답 데이터셋이 필수적이어서 새로운 도구에 대한 일반화 능력이 떨어진다.

### 2. RGB-D 기반 및 제로샷 접근 방식

- **RGB-D 방식**: 깊이 정보를 추가하여 정확도를 높이지만, 수술실 내의 근접 환경에서 발생하는 센서 사각지대 및 반사광으로 인해 하드웨어 깊이 센서 사용이 어렵다.
- **제로샷 방식**: FoundationPose나 SAM-6D와 같이 특정 객체에 대한 학습 없이 CAD 모델만으로 포즈를 추정한다. 그러나 이러한 모델들은 주로 일반적인 깊이 센서 데이터를 가정하며, RMIS와 같은 특수 환경에서의 적용 가능성은 그동안 탐구되지 않았다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

제안하는 파이프라인은 **[스테레오 이미지 $\rightarrow$ 깊이 추정 $\rightarrow$ 마스크 생성 $\rightarrow$ 제로샷 포즈 추정]**의 순서로 구성된다.

### 2. 스테레오 기반 깊이 추정 (Stereo-Based Depth Estimation)

깊이 센서의 한계를 극복하기 위해 **RAFT-Stereo** 알고리즘을 사용한다.

- **변위(Disparity) 계산**: 정렬된(Rectified) 스테레오 이미지 쌍에서 3D 상관 볼륨(Correlation volume)을 구축하고, 다층 GRU(Gated Recurrent Units)를 통해 변위 맵을 반복적으로 정교화한다.
- **깊이 계산**: 계산된 변위를 바탕으로 다음 방정식에 의해 각 픽셀의 깊이 $Z(x,y)$를 산출한다.
  $$Z(x,y) = \frac{B \cdot f}{d(x,y)}$$
  여기서 $B$는 스테레오 카메라 사이의 거리인 Baseline, $f$는 카메라의 초점 거리(Focal length), $d(x,y)$는 계산된 변위(Disparity)이다.

### 3. Mask R-CNN을 이용한 마스크 생성

수술 환경의 가려짐 문제를 해결하기 위해 미세 조정된 Mask R-CNN을 사용한다.

- **의사 라벨(Pseudo Label) 생성**: 3D CAD 모델을 2D 이미지 평면에 투영하여 초기 마스크를 생성한 후, 실제 측정된 깊이 $Z_{disp}$와 투영된 모델의 깊이 $Z_{proj}$를 비교하여 가려진 부분을 제거한다.
  $$|Z_{proj}(u,v) - Z_{disp}(u,v)| < \epsilon$$
  여기서 임계값 $\epsilon$은 $1\text{mm}$로 설정된다. 이 조건을 만족하는 픽셀만 마스크에 유지함으로써 실제 가시 영역만 추출한다.
- **학습 전략**: 합성 데이터와 실제 데이터를 혼합하여 Mask R-CNN을 미세 조정함으로써 데이터 수집 비용을 줄이고 현실적인 RMIS 시나리오에 대한 일반화 성능을 높였다.

### 4. 제로샷 RGB-D 포즈 추정

최종 단계에서는 RGB-D 이미지, 생성된 마스크, 그리고 도구의 3D CAD 모델을 입력으로 하여 포즈를 추정한다.

- 본 연구에서는 특히 **SAM-6D**의 세그멘테이션 모듈을 앞서 설명한 **Mask R-CNN**으로 교체하여 적용하였다.
- 추론 절차는 입력 데이터로부터 특징(Feature)을 추출하고, CAD 모델의 3D 구조와 2D 이미지 포인트를 정렬하여 초기 포즈 가설을 세운 뒤, 최적화를 통해 최종 포즈를 정교화하는 과정을 거친다.

## 📊 Results

### 1. 실험 설정

- **대상 도구**: Da Vinci™ Si 시스템의 Endowrist™ Large Needle Driver (LND).
- **데이터셋**:
  - Dataset A: 가려짐이 없는 상태의 이미지 (1,027장).
  - Dataset B: 다른 도구들에 의해 일부 가려진 상태의 이미지 (797장).
  - Dataset C: Mask R-CNN 학습을 위한 실제 및 합성 이미지 혼합 데이터 (총 2,489장).
- **평가 지표**:
  - 세그멘테이션: $\text{AP}$ (Average Precision), $\text{AR}$ (Average Recall).
  - 포즈 추정: $\text{ADD}$ (Average Distance of Model Points, 3D 정렬 오차), 2D Projection (이미지 평면 투영 오차).

### 2. 세그멘테이션 성능 (Dataset B 기준)

Mask R-CNN(ResNet-101)이 SAM보다 월등한 성능을 보였다. 특히 $\text{AP}@[0.50:0.95]$ 기준 Mask R-CNN은 $86.94\%$를 기록한 반면, SAM은 $46.4\%$에 그쳤다. 이는 수술 환경의 특수성으로 인해 일반적인 제로샷 세그멘테이션 모델인 SAM보다 도메인 특화 학습된 모델이 훨씬 유리함을 시사한다.

### 3. 포즈 추정 성능

- **비가려짐 상황 (Non-Occluded)**: $\text{ADD}$ 지표에서는 FoundationPose가 가장 높았으나($5\text{mm}$ 임계값 기준 $48.29\%$), 제안된 SAM-6D (Mask R-CNN) 역시 $46.86\%$로 매우 근접한 성능을 보였다. 특히 2D Projection 지표($50\text{px}$ 임계값)에서는 제안 방법이 $95.14\%$로 FoundationPose($90.29\%$)를 앞섰다.
- **가려짐 상황 (Occluded)**: 제안된 SAM-6D (Mask R-CNN)가 압도적인 성능을 보였다. $5\text{mm}$ $\text{ADD}$ 임계값 기준, 제안 방법은 $49.06\%$의 정확도를 보인 반면, FoundationPose는 $6.02\%$에 불과했다.
- **통계적 강건성**: 제안 방법의 $\text{ADD}$ 평균 오차($\mu$)는 $6.47\text{mm}$로 FoundationPose($8.05\text{mm}$)보다 낮았으며, 표준편차($\sigma$) 또한 $9.56\text{mm}$로 훨씬 작아 추정 결과가 일관적임을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 제로샷 포즈 추정 모델을 수술 환경에 적용함에 있어 **정밀한 마스크 생성**과 **신뢰할 수 있는 깊이 정보**가 성능을 결정짓는 핵심 요소임을 입증하였다.

- **강점**: 제안된 파이프라인은 특히 가려짐이 빈번한 실제 수술 환경에서 기존의 SOTA 제로샷 모델들보다 훨씬 강건한 성능을 보여준다. 또한, 합성 데이터를 활용한 학습을 통해 새로운 도구에 대해서도 추가 학습 없이 대응할 수 있는 일반화 능력을 확보하였다.
- **한계 및 논의**: 논문에서는 PVNet과 같은 지도 학습 기반 방법이 절대적인 성능은 더 높을 수 있음을 언급한다. 그러나 이는 막대한 데이터 요구량이라는 기회비용이 따른다. 제안 방법은 실시간성 측면에서 첫 프레임 초기화에 $2.52$초가 소요되지만, 이후 프레임부터는 $15\text{fps}$로 동작하여 실용적인 수준의 속도를 확보하였다.
- **비판적 해석**: 깊이 추정을 위해 RAFT-Stereo를 사용했으나, 스테레오 카메라의 캘리브레이션 상태에 따라 결과가 달라질 수 있다. 또한, Mask R-CNN의 미세 조정에 사용된 데이터셋의 다양성이 실제 수술의 모든 변수를 커버할 수 있는지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 RMIS 환경에서 새로운 수술 도구에 대해 학습 없이 포즈를 추정하는 **SurgPose** 파이프라인을 제안한다. 핵심은 스테레오 비전(RAFT-Stereo)을 통한 깊이 맵 생성과 미세 조정된 Mask R-CNN을 통한 정밀 마스크 생성으로, 이를 SAM-6D 모델에 통합하여 가려짐이 심한 환경에서도 기존 제로샷 모델(FoundationPose 등)보다 월등한 포즈 추정 정확도와 일관성을 달성하였다. 이는 향후 수술 로봇의 자율 제어 및 내비게이션 시스템의 일반화 능력을 크게 향상시킬 수 있는 연구이다.
