# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision

Utsav Rai, Haozheng Xu and Stamatia Giannarou (2025)

## 🧩 Problem to Solve

본 논문은 로봇 보조 최소 침습 수술(Robot-assisted Minimally Invasive Surgery, RMIS) 환경에서 수술 도구의 정확한 6자유도(6 Degrees of Freedom, 6DoF) 포즈 추정(Pose Estimation) 문제를 해결하고자 한다. 수술 도구의 포즈 추정은 수술 내비게이션과 로봇 제어의 정밀도 및 안전성을 확보하는 데 필수적이다.

기존의 접근 방식들은 다음과 같은 한계를 가지고 있다.

1. **마커 기반 방법(Marker-based methods):** 멸균 환경에 적합한 특수 마커가 필요하며, 가려짐(occlusion)이나 반사 표면으로 인해 정확도가 떨어진다. 또한, 케이블 구동 로봇 시스템의 경우 기구학적(kinematic) 데이터의 캘리브레이션 부족으로 인해 오차가 발생한다.
2. **지도 학습 기반 방법(Supervised learning methods):** 마커 없이 포즈를 추정할 수 있으나, 새로운 도구를 도입할 때마다 대규모의 정답 라벨링 데이터셋이 필요하다. 이는 도구가 빈번하게 교체되는 동적인 수술 환경에서 일반화(generalisation) 능력을 심각하게 제한한다.
3. **Zero-shot 포즈 추정 모델:** 최근 일반화 능력이 뛰어난 Zero-shot 모델들이 등장했으나, 이들은 주로 깊이 센서(depth sensor)에 의존한다. 하지만 수술 환경은 근접 촬영이 많아 센서의 블라인드 존(blind zones)이 발생하고 금속 도구의 강한 반사로 인해 깊이 센서 사용이 비현실적이다.

따라서 본 논문의 목표는 별도의 도구별 학습 없이도 새로운 도구에 적용 가능한 **Generalisable Zero-shot RGB-D 포즈 추정 파이프라인**을 RMIS 환경에 맞게 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 최신 Zero-shot RGB-D 포즈 추정 모델(FoundationPose, SAM-6D 등)을 RMIS 환경에 적용하기 위해, **물리적인 깊이 센서를 소프트웨어 기반의 스테레오 비전으로 대체**하고, **범용 세그멘테이션 모델을 수술 환경에 최적화된 전용 모델로 교체**하는 것이다.

주요 기여 사항은 다음과 같다.

- **비전 기반 깊이 추정 도입:** 깊이 센서 대신 RAFT-Stereo 알고리즘을 사용하여 반사가 심하고 텍스처가 없는 수술 환경에서도 강건한 깊이 지도를 생성하였다.
- **세그멘테이션 모듈 강화:** SAM-6D 모델의 기존 SAM(Segment Anything Model) 모듈을 수술 도구 데이터로 파인튜닝(fine-tuned)한 Mask R-CNN으로 교체하여, 가려짐과 반사가 심한 환경에서 마스크 생성 정확도를 크게 향상시켰다.
- **신규 데이터베이스 구축:** 실제 및 합성 스테레오 이미지와 정답 포즈 정보를 포함하는 수술 도구 포즈 추정 전용 데이터베이스를 구축하여 검증에 활용하였다.
- **종합적 성능 평가:** 다양한 Zero-shot 모델들을 RMIS 환경에서 평가하여 각 모델의 한계를 분석하고, 제안한 개선 방법의 유효성을 입증하였다.

## 📎 Related Works

논문에서는 6D 포즈 추정 방식을 크게 RGB 기반과 RGB-D 기반으로 나누어 설명한다.

1. **RGB-Based Approaches:**
   - **간접 방법(Indirect methods):** 2D 키포인트를 검출하고 PnP(Perspective-n-Point) 알고리즘을 통해 3D 모델과 매칭하는 방식이다. 가려짐에 강건하지만, 반복적인 비미분 가능 솔버(non-differentiable solvers)를 사용하므로 실시간 적응력이나 엔드-투-엔드 학습에 제약이 있다.
   - **직접 방법(Direct methods):** 입력 이미지에서 포즈를 직접 회귀(regression)하는 방식이다. 복잡한 환경에서 성능이 좋지만, 대규모 라벨링 데이터셋이 필수적이어서 새로운 도구 도입 시 재학습이 필요하다.

2. **RGB-D-Based Approaches:**
   - 깊이 데이터를 함께 사용하여 정확도를 높이며, 특히 가려짐이 심한 환경에서 유리하다. 그러나 RMIS에서는 깊이 센서의 물리적 제약(반사, 블라인드 존)으로 인해 적용이 어려웠다.
   - **Zero-shot 모델(FoundationPose, SAM-6D, OVE-6D, MegaPose):** 특정 객체에 대한 학습 없이 CAD 모델만으로 포즈를 추정한다. 본 논문은 이러한 모델들이 요구하는 '정확한 마스크'와 '깊이 정보'를 RMIS 환경에서 어떻게 확보할 것인가에 집중하여 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

제안된 전체 파이프라인은 **[스테레오 이미지 $\rightarrow$ 깊이 추정 및 마스크 생성 $\rightarrow$ Zero-shot 포즈 추정]**의 단계로 구성된다.

### 1. 스테레오 기반 깊이 추정 (Stereo-Based Depth Estimation)

물리적 깊이 센서 대신 RAFT-Stereo 알고리즘을 사용하여 깊이 지도를 생성한다.

- **Disparity 계산:** RAFT-Stereo는 Convolutional GRU를 통해 반복적으로 변위(disparity)를 정교화하며, 3D 상관 볼륨(correlation volume)을 구축하여 픽셀 간 유사성을 인코딩한다. 이는 텍스처가 없는 수술 도구 표면에서도 강건한 성능을 보인다.
- **깊이 계산:** 계산된 변위를 바탕으로 다음의 관계식을 통해 픽셀별 깊이 $Z(x,y)$를 산출한다. 여기서 $B$는 스테레오 카메라 사이의 거리(baseline)이고, $f$는 카메라의 초점 거리(focal length)이다.

### 2. Mask R-CNN을 이용한 마스크 생성

Zero-shot 세그멘테이션 모델인 SAM이 수술 환경의 반사와 가려짐에 취약하므로, 파인튜닝된 Mask R-CNN을 사용한다.

- **의사 라벨(Pseudo Label) 생성:** 3D CAD 모델을 2D 이미지 평면에 투영하여 초기 마스크를 생성한다. 이후, 투영된 모델의 깊이 $Z_{proj}(u,v)$와 RAFT-Stereo로 얻은 실제 깊이 $Z_{disp}(u,v)$를 비교하여 가려진 부분을 제거한다.
  $$|Z_{proj}(u,v) - Z_{disp}(u,v)| < \epsilon$$
  여기서 임계값 $\epsilon$은 $1\text{mm}$로 설정되며, 이 조건을 만족하는 픽셀만 마스크에 남겨 실제 가시적인 영역만 추출한다.
- **학습 데이터:** 합성 데이터(NVISII 렌더러 사용)와 실제 이미지 데이터를 혼합하여 Mask R-CNN을 학습시킴으로써 일반화 성능을 높였다.

### 3. Zero-Shot RGB-D 포즈 추정

최종 단계에서는 생성된 RGB 이미지, 깊이 지도, 도구 마스크, 그리고 3D CAD 모델을 입력으로 하여 포즈를 추정한다.

- **모델 적용:** FoundationPose, SAM-6D, OVE-6D, MegaPose 등을 평가하였다.
- **핵심 수정 사항:** 특히 SAM-6D의 경우, 내부의 인스턴스 세그멘테이션 모듈인 SAM을 위에서 설명한 **파인튜닝된 Mask R-CNN으로 교체**하여 적용하였다.

## 📊 Results

### 실험 설정

- **대상 도구:** Da Vinci™ Si 시스템의 Endowrist™ Large Needle Driver (LND).
- **데이터셋:**
  - Dataset A: 가려짐이 없는 상태의 이미지 (1,027장).
  - Dataset B: 다른 수술 도구에 의해 부분적/전체적으로 가려진 상태의 이미지 (797장).
  - Dataset C: Mask R-CNN 학습을 위한 합성 및 실제 이미지 혼합 데이터 (약 2,489장).
- **평가 지표:**
  - 세그멘테이션: $AP$ (Average Precision), $AR$ (Average Recall).
  - 포즈 추정: $ADD$ (Average Distance of Model Points), $2D \text{ Projection}$ 거리.

### 주요 결과

1. **세그멘테이션 성능:** Mask R-CNN (ResNet-101)이 SAM보다 훨씬 높은 성능을 보였다. 특히 가려짐이 있는 Dataset B에서 $\text{AP}@[IoU=0.50:0.95]$ 기준 Mask R-CNN은 $88.5\%$를 기록한 반면, SAM은 $46.4\%$에 그쳤다.
2. **포즈 추정 성능 (비가려짐 상황):** FoundationPose가 가장 우수한 성능을 보였으나, 제안한 **SAM-6D (Mask R-CNN)** 역시 $5\text{mm}$ ADD 임계값에서 $46.86\%$의 정확도를 보이며 근접한 성능을 냈다. (기존 SAM-6D는 $0.88\%$로 매우 저조함)
3. **포즈 추정 성능 (가려짐 상황):** 제안한 **SAM-6D (Mask R-CNN)**가 모든 모델 중 압도적인 성능을 보였다. $5\text{mm}$ ADD 임계값에서 $49.06\%$의 정확도를 기록했으며, 이는 FoundationPose의 $6.02\%$와 극명한 대조를 이룬다.
4. **통계적 강건성:** SAM-6D (Mask R-CNN)는 ADD의 평균($\mu$)이 $6.47\text{mm}$로 FoundationPose($8.05\text{mm}$)보다 낮았으며, 표준편차($\sigma$) 또한 $9.56\text{mm}$로 훨씬 낮아 일관된 성능을 보였다.
5. **추론 속도:** 첫 프레임 초기화에는 $2.52\text{s}$가 소요되지만, 이후 프레임부터는 약 $15\text{fps}$의 실시간 속도로 작동한다.

## 🧠 Insights & Discussion

본 연구는 Zero-shot 포즈 추정 모델을 실제 수술 환경에 적용하기 위해 필요한 핵심 요소가 **'정확한 마스크'**와 **'신뢰할 수 있는 깊이 정보'**임을 입증하였다.

- **강점:** 기존의 Zero-shot 모델들이 일반적인 환경에서는 잘 작동하지만, 수술 도구와 같은 고반사-저텍스처 환경에서는 세그멘테이션 단계부터 실패한다는 점을 정확히 짚어냈다. 이를 전용 모델(Mask R-CNN)과 스테레오 비전으로 해결함으로써, 새로운 도구에 대해서도 재학습 없이 적용 가능한 실용적인 파이프라인을 제시하였다.
- **한계 및 해석:** 결과 표에서 보듯, 가려짐이 없는 환경에서는 FoundationPose가 여전히 우세하다. 이는 FoundationPose의 내부 구조가 정밀한 정렬에 더 최적화되어 있음을 시사한다. 하지만 수술 중 빈번하게 발생하는 가려짐 상황에서 SAM-6D (Mask R-CNN)가 압도적인 성능 향상을 보인 것은, 포즈 추정 엔진 자체보다 **입력 데이터(마스크)의 품질이 결과에 더 결정적인 영향**을 미친다는 것을 의미한다.
- **비판적 시각:** 지도 학습 기반 방법(예: PVNet)이 여전히 절대적인 정확도는 더 높을 수 있다. 따라서 제안 방법은 '최고의 정확도'보다는 '새로운 도구에 대한 즉각적인 적용 가능성(Generalisability)'과 '환경적 강건성'에 가치를 둔 솔루션으로 해석해야 한다.

## 📌 TL;DR

본 논문은 수술 로봇 환경에서 새로운 도구에 대해 재학습 없이 포즈를 추정할 수 있는 **Zero-shot RGB-D 포즈 추정 파이프라인(SurgPose)**을 제안한다. 물리적 깊이 센서 대신 **RAFT-Stereo**를 통한 소프트웨어적 깊이 추정을 도입하고, SAM 대신 수술 도구에 최적화된 **Mask R-CNN**을 사용하여 세그멘테이션 정확도를 높였다. 실험 결과, 특히 도구가 가려진 복잡한 환경에서 기존 SOTA 모델인 FoundationPose를 크게 상회하는 강건성을 보였으며, 이는 향후 다양한 수술 도구를 유연하게 지원해야 하는 차세대 수술 로봇 시스템의 포즈 추정 표준이 될 가능성이 높다.
