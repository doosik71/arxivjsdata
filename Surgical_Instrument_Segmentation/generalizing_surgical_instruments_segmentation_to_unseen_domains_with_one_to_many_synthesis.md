# Generalizing Surgical Instruments Segmentation to Unseen Domains with One-to-Many Synthesis

An Wang, Mobarakol Islam, Mengya Xu, and Hongliang Ren (2023)

## 🧩 Problem to Solve

본 논문은 수술 장면 이해(Surgical Scene Understanding)를 위한 딥러닝 모델, 특히 수술 도구 분할(Surgical Instrument Segmentation) 모델이 실제 환경에 배포될 때 겪는 데이터 관련 문제를 해결하고자 한다. 구체적인 문제는 다음과 같다.

첫째, 수술 데이터의 수집과 정밀한 어노테이션(Annotation) 작업은 매우 많은 시간과 노동력이 소요되어 대규모의 학습 데이터를 확보하기 어렵다. 둘째, 수술 센터나 환자에 따라 발생하는 도메인 시프트(Domain Shift), 즉 강도 변화(Intensity shift), 획득 방식의 차이(Acquisition shift), 인구통계학적 차이(Population shift) 등으로 인해 특정 데이터셋으로 학습된 모델의 일반화 성능이 급격히 저하된다.

따라서 본 연구의 목표는 극히 적은 양의 소스 이미지(Seed images)만을 활용하여 대량의 합성 데이터셋을 생성하는 'One-to-Many Synthesis' 프레임워크를 구축하고, 이를 통해 학습되지 않은 새로운 도메인(Unseen domains)에서도 뛰어난 일반화 성능을 보이는 분할 모델을 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터 중심 AI(Data-centric AI) 관점에서 모델 구조의 변경보다는 학습 데이터의 품질과 다양성을 극대화하는 것에 집중한 것이다. 주요 기여 사항은 다음과 같다.

1. **극소량의 소스 이미지 기반 합성 프레임워크**: 단 한 장의 배경 조직 이미지와 도구당 최대 3장의 전경 이미지만을 사용하여 수동 어노테이션 없이 대규모 합성 데이터셋을 생성하는 파이프라인을 제안하였다.
2. **블렌딩 기법의 최적화**: 합성 이미지의 현실감을 높이고 경계면의 아티팩트(Artifact)를 줄이기 위해 Alpha, Gaussian, Laplacian Blending 세 가지 기법을 비교 분석하였으며, Laplacian Blending이 가장 우수한 성능을 보임을 입증하였다.
3. **하이브리드 학습 시점 증강(Hybrid Training-time Augmentation)**: 과적합을 방지하고 데이터 다양성을 확장하기 위해 Coarsely Dropout(CDO), Chained Augmentation Mixing(CAM), Element-wise Patch Mixing(EPM)을 통합한 하이브리드 증강 파이프라인을 도입하였다.
4. **강력한 일반화 성능 증명**: 합성 데이터만으로 학습한 모델이 도메인 갭이 큰 데이터셋(RoboTool)에서 실제 데이터로 학습한 모델보다 더 높은 정확도를 보임을 확인하였으며, 소량의 실제 데이터를 함께 사용하는 준지도 학습(Semi-supervised) 방식의 효율성을 제시하였다.

## 📎 Related Works

기존의 이미지 합성 기반 데이터 생성 방식인 "Cut, Paste and Learn"이나 "mix-blend" 등은 대량의 전경 및 배경 소스 이미지가 필요하다는 한계가 있었다. 또한, 최근의 Simulation-to-Real 접근 방식(예: SSIS-Seg)은 스타일 전이를 위해 타겟 도메인의 실제 데이터셋이 필요하므로, 완전히 새로운(Unseen) 도메인에 적용하기에는 제약이 따른다.

본 논문은 이러한 한계를 극복하기 위해 소스 이미지의 수를 극단적으로 제한하면서도, 강한 증강(Augmentation)과 정교한 블렌딩 기법을 결합하여 데이터 의존성을 최소화하고 일반화 능력을 높였다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. One-to-Many Dataset Synthesizing

합성 데이터 생성 과정은 크게 세 단계로 구성된다.

**가. 소스 이미지 풀 구축 (Source Image Pools Construction)**

- **배경(Background)**: 다양한 조직 모습이 포함된 단 한 장의 배경 시드 이미지 $x_b$를 준비하고, 여기에 `imgaug` 라이브러리를 통해 수평/수직 뒤집기, 크롭, 색조/채도 변경, 가우시안 블러, 노이즈 추가 등 강한 증강을 적용하여 $m$개의 이미지로 구성된 배경 풀 $X_b^m = \{x_1^b, x_2^b, ..., x_m^b\}$을 생성한다.
- **전경(Foreground)**: 도구당 최대 3장의 전경 시드 이미지 $x_f$와 이에 대응하는 마스크 $y_f$를 준비한다. 기하학적 변환(뒤집기, 아핀 변환 등)을 동일하게 적용하여 전경 이미지 풀 $X_f^n$과 마스크 풀 $Y_f^n$을 구축한다.

**나. 블렌딩 기반 이미지 합성 (Blending-based Image Composition)**
구축된 풀에서 배경과 전경 이미지를 무작위로 샘플링하여 합성 이미지를 생성한다. 본 논문은 세 가지 블렌딩 모드를 탐구한다.

- **Alpha Blending**: 전경 마스크 $y_f$ 영역은 전경 이미지를 그대로 가져오고, 나머지는 배경 이미지를 유지하는 단순 합성 방식이다.
- **Gaussian Blending**: 전경 마스크 $y_f$를 침식(Erode)시키고 블러링하여 새로운 마스크 $\tilde{y}_f$를 만든 뒤 Alpha Blending을 수행하여 경계선을 부드럽게 만든다.
- **Laplacian Blending**: 전경과 배경 이미지에 대해 라플라시안 피라미드(Laplacian pyramid)를 구축하고, 가우시안 피라미드를 가중치로 사용하여 다중 스케일에서 합성한다. 이는 경계면의 세부 구조를 유지하면서 가장 자연스러운 합성을 가능하게 한다.

**다. 하이브리드 학습 시점 증강 (Hybrid Training-time Augmentation)**
합성된 데이터 $X_{syn}^k$와 마스크 $Y_{syn}^k$를 모델에 입력하기 전, 다음 세 가지 증강을 결합하여 적용한다.

- **Coarsely Dropout (CDO)**: 이미지의 무작위 패치 영역을 삭제하여 도구의 가려짐(Occlusion) 현상을 시뮬레이션하고 과적합을 방지한다.
- **Chained Augmentation Mixing (CAM)**: 여러 개의 증강 체인(Contrast, Brightness, Sharpness 등)을 통과한 결과물들을 원본 이미지와 섞어 픽셀 수준의 다양성을 확보한다.
- **Element-wise Patch Mixing (EPM)**: 서로 다른 이미지 간의 패치를 교환하여 도구 간의 겹침(Overlapping) 현상을 구현한다.

### 2. 학습 절차 및 모델 구조

- **아키텍처**: 표준 UNet 구조를 사용한다.
- **손실 함수**: 이진 교차 엔트로피(Binary Cross-Entropy, BCE) 손실 함수를 사용하여 도구 영역과 배경 영역을 구분한다.
- **최적화**: Adam 옵티마이저를 사용하며, 학습률 $0.001$, 배치 크기 $64$, 총 $100$ 에포크(Epoch) 동안 학습한다.
- **평가 지표**: Dice Similarity Coefficient (DSC)를 사용하여 분할 성능을 측정한다.

## 📊 Results

### 1. 실험 설정

- **합성 데이터셋**: Endo18 데이터셋에서 추출한 시드 이미지를 사용하여 5가지 설정(시드 수, 이미지당 도구 수 등)의 합성 데이터셋을 구축하였다.
- **평가 데이터셋**: In-distribution 평가를 위해 Endo18 테스트셋을, Out-of-distribution(OOD) 일반화 평가를 위해 Endo17과 RoboTool 데이터셋을 사용하였다. 특히 RoboTool은 도메인 갭이 매우 커서 베이스라인 모델의 성능이 30% 이상 하락하는 어려운 데이터셋이다.

### 2. 주요 결과

- **블렌딩 기법 비교**: 실험 결과, 대부분의 설정에서 **Laplacian Blending**이 가장 높은 DSC를 기록하였다. 이는 다중 스케일 합성이 경계면의 아티팩트를 줄여 모델이 더 현실적인 특징을 학습하게 했음을 의미한다.
- **합성 데이터의 일반화 능력**: Table II 결과에 따르면, 합성 데이터로 학습한 모델(Syn-S3-F1F2)이 RoboTool 데이터셋에서 $60.79\%$ DSC를 기록하여, 실제 데이터로 학습한 베이스라인($52.44\%$)보다 $8.35\%$p 높은 성능을 보였다.
- **증강 기법의 효능**: 하이브리드 증강(CAM, CDO, EPM)을 모두 적용했을 때, 증강을 적용하지 않은 경우보다 OOD 데이터셋에서 평균 $7.12\%$에서 $14.53\%$까지 DSC가 상승하였다.
- **준지도 학습(Semi-supervised) 효과**: 합성 데이터에 아주 적은 양(10~20%)의 실제 데이터를 추가하여 공동 학습시켰을 때, 성능이 비약적으로 상승하였다. 특히 전체 합성 데이터에 20%의 실제 데이터를 추가했을 때 OOD 도메인 평균 성능이 크게 개선되었다.

## 🧠 Insights & Discussion

본 논문은 모델의 복잡도를 높이는 대신, **데이터의 생성 방식과 증강 전략**만으로도 도메인 일반화 문제를 해결할 수 있음을 보여주었다.

특히 주목할 점은 합성 데이터로 학습한 모델이 실제 데이터로 학습한 모델보다 특정 OOD 도메인(RoboTool)에서 더 좋은 성능을 냈다는 점이다. 이는 실제 데이터셋이 가진 특정 도메인의 편향(Bias)에 과적합되는 것보다, 광범위한 증강과 합성을 통해 생성된 가상의 데이터 분포를 학습하는 것이 결과적으로 더 강건한(Robust) 특징 추출기를 형성하게 했음을 시사한다.

다만, 현재의 프레임워크는 혈액(Blood)이나 연기(Smoke)와 같은 수술실의 동적인 환경 요소까지는 완벽히 재현하지 못했다는 한계가 있다. 저자들은 향후 이러한 요소를 합성 과정에 추가하고 도메인 적응(Domain Adaptation) 기법을 결합한다면 성능을 더욱 높일 수 있을 것이라고 논의하였다.

## 📌 TL;DR

본 연구는 매우 적은 수의 시드 이미지(배경 1장, 도구당 $\le 3$장)만으로 대규모 합성 데이터셋을 구축하여 수술 도구 분할 모델의 일반화 성능을 극대화하는 **One-to-Many Synthesis** 프레임워크를 제안하였다. **Laplacian Blending**과 **하이브리드 학습 증강(CAM, CDO, EPM)**을 통해 데이터의 현실감과 다양성을 확보하였으며, 그 결과 학습되지 않은 새로운 도메인(Unseen domain)에서도 실제 데이터 학습 모델을 능가하는 일반화 성능을 달성하였다. 이 연구는 데이터 부족과 도메인 시프트 문제가 심각한 의료 영상 분야에서 효율적인 데이터 생성 전략의 중요성을 입증하였다.
