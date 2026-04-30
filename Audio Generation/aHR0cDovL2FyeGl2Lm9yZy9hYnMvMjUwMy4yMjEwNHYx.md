# M2D2: Exploring General-purpose Audio-Language Representations Beyond CLAP

Daisuke Niizumi, Daiki Takeuchi, Masahiro Yasuda, Binh Thien Nguyen, Yasunori Ohishi, and Noboru Harada (2021/2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 기존의 Contrastive Language-Audio Pre-training (CLAP) 모델들이 가진 일반화 성능의 한계이다. CLAP 모델들은 오디오와 텍스트를 공통의 특징 공간(common feature space)에 정렬함으로써 오디오-텍스트 검색이나 제로샷 분류와 같은 오디오-언어 작업에서는 뛰어난 성능을 보이지만, 정작 일반적인 오디오 작업(conventional audio tasks)에서는 성능이 낮게 나타나는 경향이 있다.

이러한 현상이 발생하는 이유는 대부분의 CLAP 모델들이 지도 학습(supervised learning) 기반의 오디오 인코더를 사용하기 때문이다. 지도 학습 모델은 학습 데이터셋과 다른 도메인의 작업에 대해 취약하며, 예를 들어 AudioSet으로 학습된 모델은 유사한 환경음 분류 작업인 ESC-50에서는 잘 작동하지만, 화자 식별(speaker identification) 작업인 VoxCeleb1에서는 매우 낮은 성능을 보인다. 반면, Self-Supervised Learning (SSL) 모델들은 특정 레이블에 의존하지 않고 일반적인 오디오 특징을 학습하므로 다양한 다운스트림 작업에서 높은 일반화 성능을 보여준다.

따라서 본 논문의 목표는 SSL의 일반적인 오디오 특징 추출 능력과 CLAP의 언어 정렬 능력을 모두 갖춘 **'범용 오디오-언어 표현(general-purpose audio-language representation)'**을 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 마스킹 기반의 SSL 프레임워크인 Masked Modeling Duo (M2D)와 CLAP의 대조 학습(contrastive learning) 목적 함수를 결합하고, 이를 강력한 LLM(Large Language Model) 기반의 문장 임베딩과 연계하는 것이다.

주요 기여 사항은 다음과 같다.
- **M2D2 프레임워크 제안**: M2D의 일반화 성능과 CLAP의 시맨틱 정렬 능력을 통합하여, 범용 오디오 작업과 오디오-언어 작업을 동시에 수행할 수 있는 표현 학습 방법을 제안한다.
- **2단계 사전 학습(Two-stage Pre-training) 전략**: 막대한 계산 자원이 필요한 LLM과 대조 학습의 결합 문제를 해결하기 위해, 오디오 인코더를 먼저 학습시키고 이후 텍스트 인코더를 정렬시키는 단계적 학습 방식을 도입하였다.
- **LLM 기반 시맨틱 감독**: NLP 분야의 SOTA 문장 임베딩 모델(NV-Embed-v2 등)을 사용하여 오디오 인코더가 더 정교한 시맨틱 정보를 학습할 수 있도록 가이드하였다.
- **성능 검증**: AudioSet 파인튜닝에서 mAP 49.0의 SOTA 성능을 달성하였으며, 음악 작업 및 오디오-언어 작업에서도 최상위권의 성능을 입증하였다.

## 📎 Related Works

### 기존 연구 및 한계
1. **Conventional Audio Representation**: PANNs, AST, HTS-AT와 같은 지도 학습 모델들은 레이블 기반의 성능은 좋으나 일반화 능력이 부족하다.
2. **SSL Audio Models**: SSAST, Audio-MAE, M2D 등은 마스킹된 입력을 복원하거나 예측하는 방식으로 범용적인 특징을 학습하지만, 학습된 특징이 언어와 정렬되어 있지 않아 언어 기반 작업에 사용할 수 없다.
3. **CLAP Models**: AudioCLIP, LAION-CLAP 등은 오디오와 텍스트를 정렬하지만, 앞서 언급한 대로 오디오 인코더의 일반화 성능이 낮아 범용 오디오 작업에는 부적합하다.

### M2D2의 차별점
기존에도 SSL과 CLAP을 결합하려는 시도(예: FLAP, Cacophony)가 있었으나, M2D2는 LLM 기반의 고성능 문장 임베딩을 활용하여 시맨틱 감독을 강화하였으며, 계산 효율성을 극대화한 2단계 학습 전략을 통해 더 큰 배치 사이즈와 효율적인 학습 프로세스를 구축하였다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
M2D2는 M2D의 마스킹 예측 목적 함수와 CLAP의 대조 학습 목적 함수를 결합한 구조이다. 특히 계산 비용을 줄이기 위해 다음과 같은 2단계 학습 과정을 거친다.

#### 1단계: 오디오 인코더 및 프로젝터 학습 (First Stage)
이 단계의 목표는 강력한 LLM 기반 텍스트 임베딩을 가이드 삼아 범용적이면서도 시맨틱한 오디오 특징을 학습하는 것이다.
- **입력**: 오디오 데이터와 미리 추출된 LLM 기반 텍스트 임베딩.
- **학습 과정**: M2D의 SSL 학습과 CLAP의 대조 학습을 동시에 수행한다. LLM 인코더는 고정(frozen)된 상태로 사용하며, 오디오 인코더와 오디오 프로젝터만 학습시킨다.
- **오디오 프로젝터**: 단순한 MLP가 아닌 Transformer Encoder를 사용하여, 오디오 특징 $z_v$를 더 정교한 시맨틱 특징 $s_a$로 요약한다.

#### 2단계: 텍스트 인코더 학습 (Second Stage)
1단계에서 학습된 고성능 오디오 특징을 유지하면서, 이에 최적화된 텍스트 인코더를 학습시키는 단계이다.
- **입력**: 오디오 데이터와 텍스트 캡션.
- **학습 과정**: 오디오 인코더는 고정하고, 텍스트 인코더(예: BERT)와 오디오 프로젝터를 학습시킨다.
- **목적**: LLM의 지식이 전이된 오디오 특징에 맞춰 텍스트 인코더가 정렬되도록 하여, 최종적으로 효율적인 CLAP 특징 쌍을 생성한다.

### 주요 방정식 및 손실 함수

#### 1. M2D 손실 함수 ($\mathcal{L}_{m2d}$)
M2D는 온라인 네트워크가 타겟 네트워크의 마스킹된 특징을 예측하도록 학습한다. $\ell_2$-정규화된 예측값 $\hat{z}_m$과 실제 타겟 값 $\tilde{z}_m$ 사이의 평균 제곱 오차(MSE)를 사용한다.
$$\mathcal{L}_{m2d} \triangleq ||\ell_2(\hat{z}_m) - \ell_2(\tilde{z}_m)||_2^2 = 2 - 2 \cdot \frac{\langle \hat{z}_m, \tilde{z}_m \rangle}{||\hat{z}_m||_2 \cdot ||\tilde{z}_m||_2}$$

#### 2. CLAP 손실 함수 ($\mathcal{L}_{clap}$)
오디오 시맨틱 특징 $s_a$와 텍스트 시맨틱 특징 $s_t$ 사이의 코사인 유사도 $S_{mn}$를 기반으로 NT-Xent loss를 계산한다.
$$S_{mn} = \frac{\langle s_a^{(m)}, s_t^{(n)} \rangle}{||s_a^{(m)}||_2 \cdot ||s_t^{(n)}||_2}$$
$$\mathcal{L}_{clap} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{B} \exp(S_{ji}/\tau)} + \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{B} \exp(S_{ij}/\tau)} \right]$$
여기서 $B$는 배치 사이즈, $\tau$는 학습 가능한 온도 파라미터이다.

#### 3. 전체 통합 손실 함수 (Stage 1)
$$\mathcal{L}_{stage1} = \lambda_{m2d} \mathcal{L}_{m2d} + \lambda_{clap} \mathcal{L}_{clap}$$
실험적으로 $\lambda_{m2d} = 1.0$, $\lambda_{clap} = 0.01$로 설정하여 CLAP의 강력한 신호가 SSL 학습을 방해하지 않도록 조절하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: AudioSet, VGGSound, WavCaps (사전 학습), AudioCaps, Clotho (평가).
- **평가 작업**:
    - **범용 오디오 작업**: ESC-50, UrbanSound8K, VoxCeleb1 등 (Linear Evaluation 및 Fine-tuning).
    - **음악 작업**: MARBLE 벤치마크 (GTZAN, NSynth 등).
    - **오디오-언어 작업**: 제로샷 분류, 오디오-텍스트 검색 (R@k, mAP), 오디오 캡셔닝 (METEOR, CIDEr 등).

### 주요 결과
1. **범용 오디오 성능 (Linear Evaluation)**:
    - M2D2는 VoxCeleb1(화자 식별)과 같은 작업에서 기존 CLAP 모델들보다 압도적으로 높은 성능을 보였다. 이는 SSL의 결합이 소리 자체의 고유한 특징을 학습하는 데 기여했음을 보여준다.
    - M2D2-AS+ (AudioSet 파인튜닝 모델)는 ESC-50에서 97.9%, US8K에서 89.7%로 SOTA 성능을 달성하였다.

2. **음악 작업 성능**:
    - MARBLE 벤치마크에서 SOTA 음악 전용 모델들과 대등하거나 더 높은 성능을 보였으며, 특히 M2D2-AS+는 일부 지표에서 새로운 SOTA를 기록하였다.

3. **오디오-언어 작업 성능**:
    - **제로샷 분류**: GTZAN에서 79.31%로 SOTA를 달성하는 등 전반적으로 최상위권 성능을 보였다.
    - **오디오-텍스트 검색**: 2단계 학습을 거친 M2D2가 1단계 모델(M2D2-)보다 훨씬 높은 R@1 성능을 보여, 텍스트 인코더의 최적화가 세밀한 정렬에 필수적임을 입증하였다.
    - **오디오 캡셔닝**: EnCLAP 프레임워크 내에서 M2D2-AS+가 AudioCaps 데이터셋에 대해 최상위 성능을 기록하였다.

## 🧠 Insights & Discussion

### 분석 및 강점
- **SSL과 CLAP의 상호보완성**: 본 연구는 SSL이 소리의 물리적/구조적 특징을 학습하게 하고, CLAP이 이를 고수준 시맨틱 공간으로 매핑하게 함으로써 두 마리 토끼를 모두 잡을 수 있음을 보여주었다.
- **LLM의 역할**: NLP 분야의 고성능 LLM 임베딩을 사용한 것이 오디오 인코더의 시맨틱 학습 효율을 극대화하였다. 이는 오디오-언어 모델 학습 시 텍스트 쪽의 강력한 사전 학습 지식을 가져오는 것이 매우 유효함을 시사한다.
- **학습 전략의 효율성**: 2단계 학습 방식을 통해 GPU 메모리 문제를 해결하고, 배치 사이즈를 2048까지 확장하여 대조 학습의 성능을 높였다.

### 한계 및 논의사항
- **성능 트레이드-오프**: 실험 결과, CLAP 목적 함수를 너무 강하게 적용하면($\lambda_{clap}$ 값이 높으면) 범용 오디오 성능이 저하되는 현상이 발견되었다. 이는 시맨틱 정렬과 일반적 특징 학습 사이에 어느 정도의 트레이드-오프가 존재함을 의미한다.
- **데이터 의존성**: 특정 벤치마크(예: AudioCaps)에서는 AudioSet 파인튜닝이 필수적이었으나, 다른 데이터셋(예: Clotho)에서는 오히려 일반화 성능이 더 중요하게 작용하는 등 데이터셋의 성격에 따른 성능 차이가 존재한다.

## 📌 TL;DR

M2D2는 SSL 기반의 M2D와 대조 학습 기반의 CLAP을 결합하고, LLM의 강력한 문장 임베딩을 활용하여 **범용 오디오 특징과 시맨틱 정렬 특징을 동시에 제공하는 표현 학습 모델**이다. 2단계 학습 전략을 통해 계산 효율성을 확보했으며, 결과적으로 기존 CLAP 모델들이 가지지 못했던 일반 오디오 작업의 일반화 성능과 최상위 수준의 오디오-언어 정렬 성능을 모두 달성하였다. 이 연구는 향후 오디오-언어 모델이 단순한 분류를 넘어 정밀한 오디오 분석 및 이해로 나아가는 데 중요한 기초를 제공한다.