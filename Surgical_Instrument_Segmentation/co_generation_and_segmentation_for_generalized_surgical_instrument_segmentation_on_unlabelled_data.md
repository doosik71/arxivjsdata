# Co-Generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data

Megha Kalia, Tajwar Abrar Aleef, Nassir Navab, and Septimiu E. Salcudean (2021)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술(Robot-Assisted Surgery, RAS)에서 증강 현실(AR) 오버레이 및 정확한 도구 추적을 위해 필수적인 수술 도구 분할(Surgical Instrument Segmentation) 문제를 다룬다. 최근 딥러닝 기반 방법론들이 뛰어난 성능을 보이고 있으나, 이러한 모델들은 대량의 라벨링된 데이터에 의존한다는 치명적인 한계가 있다. 수술 데이터의 특성상 정밀한 라벨링 작업은 비용과 시간이 많이 소요되며, 이로 인해 실제 임상 환경으로의 기술 이전(Surgical Translation)에 병목 현상이 발생한다.

특히, 기존 방법론들은 학습 데이터와 다른 도메인의 데이터(예: 동물 실험 데이터로 학습 후 실제 인간 수술 데이터에 적용)에 대해 일반화 성능(Generalization capability)이 현저히 떨어진다는 문제가 있다. 따라서 본 논문의 목표는 라벨링되지 않은 타겟 도메인의 데이터가 존재할 때, 라벨링된 소스 도메인의 데이터를 활용하여 타겟 도메인에서도 높은 성능을 내는 일반화된 분할 모델을 학습시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 생성 모델(Generative Model)과 분할 모델(Segmentation Model)을 분리하여 순차적으로 학습시키는 것이 아니라, **두 모델을 동시에 학습시키는 공동 피드백 시스템(Joint Feedback System)**을 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **coSegGAN 제안**: 생성 모델이 소스 도메인에서 타겟 도메인으로 이미지를 변환하는 동시에, 분할 모델이 생성된 이미지를 통해 학습하며 다시 생성 모델을 정규화하는 상호 보완적 학습 구조를 설계하였다.
2. **형태 보존 제약 조건 추가**: 기존 CycleGAN 등이 이미지 변환 과정에서 도구의 형태를 왜곡시키는 문제를 해결하기 위해, 분할 모델의 피드백을 이용한 Shape Loss와 잠재 공간(Latent Space)에서의 Structure Loss를 도입하였다.
3. **실제 임상 데이터 검증**: 공개 데이터셋뿐만 아니라 실제 전립선 절제술(Prostatectomy) 영상 데이터를 사용하여 제안 방법론의 일반화 성능을 입증하였다.

## 📎 Related Works

기존에는 부족한 임상 라벨링 데이터를 보완하기 위해 GAN 기반의 데이터 증강 기법들이 제안되었다. 예를 들어, 시뮬레이션 데이터를 실제 데이터로 변환하거나, 사체(Cadaver) 수술 데이터를 생체(In vivo) 데이터로 변환하여 분할 모델을 학습시키는 방식이 있었다.

그러나 이러한 기존 접근 방식들은 다음과 같은 한계가 있다.

- **형태 왜곡**: 이미지-투-이미지(I2I) 변환 과정에서 수술 도구의 기하학적 형태가 변하거나 아티팩트(Artefact)가 생성되는 경우가 많으며, 이는 임상 적용 시 치명적이다.
- **평가 지표의 부재**: 변환된 데이터의 품질을 정량적으로 측정할 수 있는 검증된 지표가 부족하다.
- **순차적 학습**: 대개 생성 모델을 먼저 학습시켜 데이터를 생성한 뒤 분할 모델을 학습시키므로, 생성 모델이 분할 모델의 요구 사항을 반영하여 최적화되지 않는다.

## 🛠️ Methodology

### 전체 시스템 구조

coSegGAN은 CycleGAN 아키텍처를 기반으로 하며, 두 개의 생성기($G_A, G_B$), 두 개의 판별기($D_A, D_B$), 그리고 하나의 분할 모델($S$)로 구성된다.

- $G_A: x_b \to x_a$ (타겟 도메인 $\psi_B$에서 소스 도메인 $\psi_A$로 변환)
- $G_B: x_a \to x_b$ (소스 도메인 $\psi_A$에서 타겟 도메인 $\psi_B$로 변환)
- $S$: 이미지 내의 수술 도구를 분할하는 U-Net 기반 모델

### 학습 절차

학습은 생성기, 판별기, 분할 모델을 교대로 업데이트하는 방식으로 진행된다.

1. **생성기 업데이트**: 판별기와 분할 모델의 가중치를 동결한 상태에서 $G_A, G_B$를 업데이트한다.
2. **판별기 및 분할 모델 업데이트**: 생성기의 가중치를 동결하고 $D_A, D_B$ 및 $S$를 업데이트한다. 이때 분할 모델 $S$는 원본 소스 이미지($x_a$)와 생성된 타겟 이미지($G_B(x_a)$)를 모두 입력으로 받아 동일한 라벨($y_a$)을 사용하여 학습한다.

### 손실 함수 (Loss Functions)

#### 1. 분할 모델 손실 ($L_{seg}$)

배경 픽셀이 압도적으로 많은 클래스 불균형 문제를 해결하기 위해 $\alpha$-balanced Focal Loss를 사용한다.
$$L_{seg} = L_{foc}(x_a, y_a) + L_{foc}(G_B(x_a), y_a)$$

#### 2. 생성 모델 손실 ($L_{generator}$)

생성기는 단순한 픽셀 일치도를 넘어 도메인 불변의 구조적 특성을 유지해야 한다.

- **Shape Loss ($L_{shape}$)**: 분할 모델 $S$의 피드백을 직접적으로 이용하며, 생성된 이미지 $G_B(x_a)$가 원래의 라벨 $y_a$와 일치하도록 강제하여 도구의 형태 왜곡을 방지한다.
$$L_{shape} = L_{foc}(G_B(x_a), y_a)$$
- **Structure Loss ($L_{structure}$)**: 생성기 내부의 인코더($e_A, e_B$)를 통해 추출된 특성 맵(Feature Map) 간의 $L_1$ 거리를 최소화하여 고수준의 시맨틱 구조를 보존한다.
$$L_{structure} = \mathbb{E}[\|e_A(x_a) - e_B(G_B(x_a))\|_1] + \mathbb{E}[\|e_B(x_b) - e_A(G_A(x_b))\|_1]$$
- **최종 생성 손실**:
$$L_{generator} = \lambda_1 L_{GAN}^{Total} + \lambda_2 L_{cyc}^{Total} + \lambda_3 L_{shape} + \lambda_4 L_{structure} + \lambda_5 L_I$$
여기서 $L_{GAN}$은 적대적 손실, $L_{cyc}$는 Cycle-consistency 손실, $L_I$는 Identity mapping 손실을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **Endovis**: 돼지 수술 데이터 (In-vivo)
  - **UCL**: 동물 조직 배경 데이터 (Ex-vivo)
  - **Surgery**: 실제 인간 전립선 절제술 데이터 (Unlabelled)
- **비교 대상**: Ternausnet, RASnet 및 CycleGAN을 이용해 데이터를 먼저 증강한 후 학습시킨 모델들(RASnet+, Ternausnet+, $\text{U-Net}_{FL}^+$)과 비교하였다.
- **평가 지표**: Mean Dice Score 및 라벨링 도메인과 비라벨링 도메인 간의 성능 차이인 $\Delta \text{Dice}$를 측정하였다. $\Delta \text{Dice}$가 낮을수록 일반화 성능이 높음을 의미한다.

### 주요 결과

- **정량적 결과**: 모든 케이스에서 coSegGAN이 타겟 도메인(Unlabelled)에 대해 가장 높은 Dice Score를 기록하였다. 특히 Case 1 (Endovis $\to$ Surgery)에서 coSegGAN의 $\Delta \text{Dice}$는 $0.9\%$로 매우 낮아, 소스 도메인과 타겟 도메인 간의 성능 차이가 거의 없음을 보였다.
- **Ablation Study**: Structure Loss를 제거한 모델($\text{coSegGAN}^-$)보다 $L_{structure}$를 포함한 모델이 특히 Endovis 데이터셋에 대해 약 $5\%$ 더 높은 성능을 보였으며, 전반적인 일반화 능력이 향상되었다.
- **정성적 결과**: 시각적 분석 결과, coSegGAN은 기존 방법론들에 비해 도구의 세밀한 구조를 더 잘 보존하며 False Positive(오검출)가 적게 발생함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 생성 모델과 분할 모델을 단일 피드백 루프 내에서 결합함으로써, 데이터 증강의 품질을 분할 성능이라는 명확한 목표에 맞춰 최적화하였다. 특히 $L_{shape}$와 $L_{structure}$의 도입은 GAN의 고질적인 문제인 형태 왜곡을 효과적으로 억제하여 임상적으로 유의미한 결과를 도출하였다.

### 한계 및 비판적 해석

- **특정 환경에서의 실패**: 정성적 분석 결과, 출혈(Blood)이 심하거나 조명이 어두운 영역에서는 도구를 제대로 식별하지 못하는 경우가 발생하였다. 이는 모델이 조명 변화나 혈액으로 인한 가려짐(Occlusion)에 대해 여전히 취약함을 시사한다.
- **데이터 다양성 부족**: UCL 데이터셋에서 학습했을 때 $\Delta \text{Dice}$가 상대적으로 높게 나타났는데, 이는 UCL 데이터가 Ex-vivo 데이터로서 실제 수술 환경의 조명 및 배경을 충분히 반영하지 못하고, 도구의 종류가 단일하여 매핑 능력이 제한되었기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 라벨링되지 않은 수술 데이터에 대한 일반화 성능을 높이기 위해, 생성 모델과 분할 모델을 동시에 학습시키는 **coSegGAN**을 제안하였다. 분할 모델의 피드백을 생성 과정에 직접 반영하는 Shape 및 Structure Loss를 통해 도구의 형태 왜곡 문제를 해결하였으며, 이를 통해 실제 인간 수술 영상에서도 높은 분할 정확도를 달성하였다. 이 연구는 라벨링 데이터 획득이 어려운 의료 AI 분야에서 비라벨링 데이터를 효과적으로 활용할 수 있는 실용적인 프레임워크를 제공한다.
