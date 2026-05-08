# Exploring optical flow inclusion into nnU-Net framework for surgical instrument segmentation

Marcos Fernández-Rodríguez, Bruno Silva, Sandro Queirós, Helena R. Torres, Bruno Oliveira, Pedro Morais, Lukas R. Buschle, Jorge Correia-Pinto, Estevão Lima, and João L. Vilaça (2024)

## 🧩 Problem to Solve

복강경 수술(Laparoscopy)에서의 수술 도구 분할(Surgical instrument segmentation)은 추적, 단계 인식, 포즈 추정 등 컴퓨터 보조 수술 시스템을 구축하는 데 필수적인 기초 작업이다. 최근 딥러닝의 발전에도 불구하고, 복강경 수술 환경의 동적인 특성으로 인해 정밀한 분할을 달성하는 것은 여전히 어려운 과제로 남아 있다.

특히, 의료 영상 분할 분야에서 뛰어난 성능을 보이는 nnU-Net 프레임워크는 단일 프레임의 정적 이미지 분석에 최적화되어 있어, 비디오 데이터가 가진 시간적 정보(Temporal information)를 활용하지 못한다는 한계가 있다. 본 논문의 목표는 Optical Flow(OF) 맵을 nnU-Net의 추가 입력 채널로 활용하여, 아키텍처의 큰 변경 없이 시간적 정보를 간접적으로 주입함으로써 수술 도구 분할 성능을 향상시키는 것이다. 이는 수술 필드 내에서 움직이는 주된 객체가 수술 도구라는 직관에 근거한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 nnU-Net의 자동 설정 능력과 접근성을 유지하면서, Optical Flow를 추가적인 모달리티(Modality)로 입력하여 성능을 높이는 것이다.

가장 중심적인 설계 전략은 복잡한 아키텍처 수정 대신, 입력 단계에서 RGB 이미지와 함께 서로 다른 표현 방식의 Optical Flow 맵을 결합하는 방식을 취한 것이다. 이를 통해 모델이 도구의 정적 외형뿐만 아니라 움직임의 특성(Temporal component)을 함께 학습하도록 유도하였다.

## 📎 Related Works

논문에서는 자가 설정형 딥러닝 분할 방법론인 nnU-Net을 주요 베이스라인으로 언급한다. nnU-Net은 데이터셋의 특성에 맞춰 자동으로 구성되어 많은 챌린지에서 state-of-the-art 성능을 기록하였으며, 낮은 전문성 요구치와 효율적인 GPU 메모리 사용량으로 인해 비교 기준 프레임워크로 널리 사용되고 있다.

기존의 일부 연구들은 nnU-Net의 아키텍처 자체를 수정(Residual, Dense, Inception 블록 추가 또는 Transformer 도입 등)하여 성능을 높이려 시도하였다. 그러나 본 연구는 아키텍처 수정보다는 입력 데이터의 풍부함을 더하는 방식에 집중하여, nnU-Net이 가진 범용성과 접근성이라는 장점을 보존하면서 시간적 정보를 통합하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 RGB 이미지와 Optical Flow 맵을 동시에 입력으로 받는 2D nnU-Net 구조를 사용한다. 전체 과정은 크게 Optical Flow 생성, 데이터 전처리, 그리고 nnU-Net 학습의 세 단계로 구성된다.

### Optical Flow 생성 및 표현 방식

Ground Truth OF 맵이 부재하므로, 비지도 학습 방식인 ARFlow를 사용하여 OF를 생성하였다. Cholec80 데이터셋으로 사전 학습된 ARFlow 모델을 사용하여 CholecSeg8k 데이터셋의 프레임 간 움직임을 추정하였다.

움직임을 추정하기 위한 시간적 간격은 두 가지 설정으로 실험되었다:

1. 현재 프레임과 직전 프레임의 비교 ($t_1$)
2. 현재 프레임과 5번째 전 프레임의 비교 ($t_5$)

또한, Optical Flow를 네트워크에 입력하기 위해 세 가지 서로 다른 표현 방식을 정의하였다:

- **RGBof**: 움직임을 색상 휠(Color wheel)로 표현한 RGB 이미지 형태
- **XY**: $x$축 및 $y$축의 변위 맵(Displacement maps)
- **PC**: 크기(Magnitude)와 각도(Angle)로 표현한 극좌표(Polar) 형태

### 학습 절차 및 제약 사항

nnU-Net의 기본 파이프라인을 사용하되, Optical Flow 정보의 무결성을 유지하기 위해 데이터 증강(Augmentation) 전략을 수정하였다. Gaussian noise, Blur, 밝기/대비 변경 등 픽셀 값에 직접적인 영향을 주는 증강은 제거하고, 회전, 스케일링, 미러링, 탄성 변형(Elastic deformations)과 같은 **기하학적 증강(Geometric augmentations)**만을 적용하였다.

입력 데이터는 nnU-Net이 서로 다른 이미지 모달리티로 처리할 수 있도록 채널별로 분리하여 입력되었으며, 모든 OF 맵은 데이터셋의 원래 해상도($856 \times 480$)로 리스케일링되었다.

### 평가 지표

모델의 성능은 Dice Coefficient (DC), Recall, Precision을 통해 측정되었으며, 각 도구 클래스(Grasper, L-hook)별 결과와 전체 평균(Mean)을 산출하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: CholecSeg8k (8,080 프레임)
- **클래스**: Background, Grasper, L-hook (총 3개 클래스)
- **비교 대상**: RGB 전용 베이스라인 vs RGB + 다양한 OF 조합 (총 7개 모델)
- **검증 방식**: 모델별 4회 학습 후 결과 평균값 사용

### 정량적 결과

실험 결과, 모든 Optical Flow 변형 모델이 RGB 베이스라인보다 우수한 성능을 보였다. 평균 Dice Coefficient (Mean DC)는 약 $+7.80\%$ 향상되었으며, Recall과 Precision은 각각 $+10.59\%$, $+8.03\%$ 증가하였다.

특히 도구별로 분석했을 때 성능 향상의 폭이 다르게 나타났다:

- **Grasper**: DC 기준 평균 $+1.07\%$의 완만한 향상을 보였다.
- **L-hook**: 매우 큰 폭의 성능 향상이 관찰되었다. 특히 **RGBof** 변형을 사용했을 때 DC $+17.71\%$, Recall $+23.78\%$, Precision $+16.94\%$의 비약적인 상승을 기록하였다.

### 변형별 분석

- **시간적 간격 ($t_1$ vs $t_5$)**: 두 설정 간의 유의미한 차이는 거의 없었으며, L-hook의 Recall에서 $t_5$가 약 $2.04\%$ 높게 나타난 수준이었다.
- **표현 방식 (RGBof vs XY vs PC)**: 일반적으로 RGBof 방식이 가장 우수한 성능을 보였으며, 특히 L-hook 클래스에서 강세를 보였다. 반면 Grasper 클래스에서는 XY 표현 방식이 약간 더 우세하였다.

## 🧠 Insights & Discussion

### 분석 및 해석

본 연구의 가장 중요한 발견은 **움직임이 많고 데이터셋 내 빈도가 낮은 클래스일수록 Optical Flow의 이점이 크다**는 점이다. L-hook은 Grasper보다 데이터셋 내 출현 빈도가 낮지만(약 $29\%$), 움직임은 더 활발한 특성이 있다. 정적인 이미지 정보만으로는 부족한 특징을 Optical Flow가 보완함으로써, 소수 클래스인 L-hook의 검출 능력이 획기적으로 개선된 것으로 해석된다.

### 한계 및 비판적 논의

1. **데이터셋 품질 문제**: 분석 과정에서 CholecSeg8k 데이터셋의 레이블 불일치(Label inconsistency)가 다수 발견되었다. 레이블 누락, 경계면의 모호함, 서로 다른 클래스의 혼용, 장기가 도구로 레이블링 된 사례 등이 확인되었으며, 이는 모델 학습의 상한선을 제한하는 요인이 되었을 가능성이 크다.
2. **데이터 증강의 제한**: OF 맵의 특성을 보존하기 위해 기하학적 증강만 사용함으로써, nnU-Net이 원래 제공하는 강력한 증강 파이프라인의 이점을 충분히 활용하지 못했다.
3. **OF의 모호성**: Optical Flow는 단순히 픽셀의 움직임을 추정하므로, 움직이는 대상이 수술 도구인지 아니면 도구에 의해 끌려가는 조직(Tissue)인지 구분하지 못한다.

## 📌 TL;DR

본 논문은 복강경 수술 도구 분할을 위해 nnU-Net 프레임워크에 Optical Flow(OF)를 추가 입력 모달리티로 통합하는 방법을 제안한다. 아키텍처의 변경 없이 RGB 이미지와 OF 맵을 함께 입력하는 것만으로도 전체적인 분할 성능이 향상되었으며, 특히 데이터 수가 적고 움직임이 많은 도구(L-hook)의 성능이 비약적으로 상승함을 입증하였다. 이 연구는 시간적 정보가 소수 클래스의 식별력을 높이는 데 효과적임을 보여주며, 향후 의료 영상 분할에서 움직임 기반 특징 추출의 중요성을 시사한다.
