# FloWaveNet : A Generative Flow for Raw Audio

Sungwon Kim, Sang-gil Lee, Jongyoon Song, Jaehyeon Kim, Sungroh Yoon

## 🧩 Problem to Solve

기존의 WaveNet 보코더는 높은 품질의 오디오를 생성하지만, 순차적 샘플링(ancestral sampling) 방식 때문에 추론 시간이 길어 실시간 애플리케이션에 한계가 있었습니다. 최근 제안된 Parallel WaveNet 및 ClariNet은 역 자기회귀 흐름(Inverse Autoregressive Flow, IAF)을 통합하여 실시간 오디오 합성을 가능하게 했으나, 이들은 다음과 같은 제한점을 가집니다:

- 잘 훈련된 교사 네트워크(teacher network)를 사용하는 2단계 훈련 파이프라인이 필요합니다.
- 보조 손실 항(auxiliary loss terms)과 결합된 확률 밀도 증류(probability distillation)를 통해서만 자연스러운 사운드를 생성할 수 있습니다 (그렇지 않으면 모드 붕괴(mode collapse) 문제가 발생하기 쉽습니다).
  이러한 문제들을 해결하기 위해, 논문은 단일 훈련 단계와 단일 최대 우도 손실(maximum likelihood loss)만을 사용하여 안정적이고 실시간으로 고품질의 원시 오디오를 합성할 수 있는 모델을 제안합니다.

## ✨ Key Contributions

- **FloWaveNet 제안**: 기존의 2단계 접근 방식과 달리 단일 최대 우도 손실과 종단 간(end-to-end) 단일 단계 훈련만을 요구하는 병렬 파형 음성 합성을 위한 새로운 흐름 기반(flow-based) 접근 방식인 FloWaveNet을 제안합니다.
- **훈련 간소화 및 안정성 입증**: 보조 손실 항 없이 2단계 접근 방식이 현실적인 오디오를 생성하기 어려운 반면, FloWaveNet의 훈련이 반복 전반에 걸쳐 크게 단순화되고 안정적임을 보여줍니다.
- **오픈 소스 구현 및 비교 연구**: FloWaveNet과 공개적으로 사용 가능한 구현보다 뛰어난 Gaussian IAF의 오픈 소스 구현을 제공하며, 공개 음성 데이터셋을 사용하여 이전 병렬 합성 모델들과의 첫 번째 비교 연구를 수행합니다.

## 📎 Related Works

- **WaveNet (Van Den Oord et al., 2016)**: 원시 오디오의 확률 분포를 추정하는 생성 모델로, 최첨단 충실도의 음성을 합성합니다. 조건부 확률의 장기 의존성을 모델링하기 위해 인과적 팽창 컨볼루션(causal dilated convolutions)을 사용합니다. 하지만 추론 시 각 타임스텝마다 모델의 전체 순방향 패스(forward pass)를 실행해야 하는 자기회귀적(autoregressive) 샘플링 때문에 실시간 오디오 생성에 주요 병목이 됩니다.
- **Parallel WaveNet (Van Den Oord et al., 2017)**: 자기회귀적 샘플링의 느린 속도를 극복하기 위해 제안되었으며, 역 자기회귀 흐름(IAF)을 통합하여 병렬 오디오 합성을 통해 실시간 기능을 달성했습니다. 그러나 KL 발산에 대한 몬테카를로(Monte Carlo) 추정 때문에 불안정하며, 모드 붕괴를 방지하기 위해 추가 보조 손실(예: 스펙트럼 거리 손실, 지각 손실, 대비 손실)이 필요합니다.
- **ClariNet (Ping et al., 2018)**: Parallel WaveNet의 몬테카를로 근사의 불안정성을 완화하기 위해 단일 가우시안 분포와 폐쇄형(closed-form) 쿨백-라이블러(KL) 발산을 사용하는 대안적인 공식을 제안했습니다. 또한 수치적 안정성을 위해 KL 발산을 정규화했습니다. 하지만 이 모델 역시 현실적인 사운드를 합성하기 위해 정규화된 KL 발산과 스펙트로그램 프레임 손실 같은 보조 손실 항을 필요로 합니다.

## 🛠️ Methodology

FloWaveNet은 흐름 기반(flow-based) 생성 모델이며, 최상위 추상 모듈인 컨텍스트 블록(context block)과 그 안에 여러 가역적 변환으로 구성된 계층적 아키텍처를 가집니다.

- **흐름 기반 생성 모델의 원리**:

  - 파형 오디오 신호 $x$를 알려진 사전 분포(prior distribution) $P_Z$로 직접 매핑하는 가역 변환 $f(x) : x \rightarrow z$가 존재한다고 가정합니다.
  - 변수 변환 공식(change of variables formula)을 사용하여 $x$의 로그 확률 분포를 명시적으로 계산합니다:
    $$ \log P_X(x) = \log P_Z(f(x)) + \log \det\left(\frac{\partial f(x)}{\partial x}\right) $$
  - 효율적인 훈련 및 샘플링을 위해 (i) 변환 $f$의 야코비안 행렬식(Jacobian determinant) 계산이 다루기 쉽고, (ii) 역변환 $x = f^{-1}(z)$를 통한 $z$에서 $x$로의 매핑이 효율적이어야 합니다.
  - 이를 위해 Real NVP (Dinh et al., 2016)에서 제안된 **아핀 커플링 레이어(affine coupling layers)**를 사용합니다.
  - **컨텍스트(조건부) 정보 활용**: 멜 스펙트로그램 $c$를 네트워크에 조건부 정보로 입력하여 조건부 확률 $p(x|c)$를 추정합니다.

- **아핀 커플링 레이어 ($f_{AC}$)**:

  - 입력 $x$를 $x_{odd}$와 $x_{even}$으로 채널을 분할합니다.
  - 변환은 다음과 같이 정의됩니다:
    $$ y*{odd} = x*{odd} $$
        $$ y*{even} = \frac{x*{even} - m(x*{odd}, c*{odd})}{\exp(s(x*{odd}, c*{odd}))} $$
  - 여기서 $m$과 $s$는 비인과적 WaveNet 아키텍처를 공유하며, $c_{odd}$는 멜 스펙트로그램 조건입니다.
  - 역변환은 효율적으로 계산 가능하며, 야코비안 행렬식은 $\log \det\left(\frac{\partial f_{AC}(x)}{\partial x}\right) = -\sum_{even} s(x_{odd}, c_{odd})$로 계산됩니다.
  - 각 흐름 작업 후에는 `Change Order` 연산이 $y_{odd}$와 $y_{even}$의 순서를 교환하여 다음 흐름에서 모든 채널이 서로 영향을 미치도록 합니다.

- **컨텍스트 블록 (Context Block)**:

  - FloWaveNet의 가장 높은 추상화 모듈입니다.
  - **스퀴즈(Squeeze) 연산**: 데이터 $x$와 조건 $c$를 받아 시간 차원 $T$를 절반으로 분할하여 채널 차원 $C$를 두 배로 만듭니다. 이는 WaveNet 기반 흐름의 유효 수용장(receptive field)을 두 배로 늘려, 상위 블록이 오디오의 장기적 특성을 학습하고 하위 블록이 고주파 정보에 집중할 수 있도록 돕습니다.
  - 블록 내 흐름 작업은 활성화 정규화(ActNorm), 아핀 커플링 레이어, 순서 변경(Change Order)으로 구성됩니다.
  - **다중 스케일 아키텍처(multi-scale architecture)**: 여러 컨텍스트 블록 이후에 피처 채널의 절반을 가우시안으로 모델링하여 분리하고, 나머지 채널은 후속 컨텍스트 블록을 거칩니다.

- **활성화 정규화 (ActNorm Layer, $f_{AN}$)**:

  - Glow (Kingma & Dhariwal, 2018)에서 제안된 방식으로, 다중 흐름 연산으로 구성된 네트워크의 훈련을 안정화합니다.
  - 흐름 시작 부분에 채널별 파라메트릭 아핀 변환을 적용합니다: $f_{AN}(x_i) = x_i \cdot s_i + b_i$.
  - 로그-디터미넌트: $\log \det\left(\frac{\partial f_{AN}(x)}{\partial x}\right) = T \cdot \sum_{i=1}^{C} \log|s_i|$.
  - 첫 번째 배치 데이터에 대해 활성화를 채널별로 평균 0, 단위 분산이 되도록 스케일링하여 $s$와 $b$를 데이터 의존적으로 초기화합니다.

- **훈련 상세**:
  - LJSpeech 데이터셋(22kHz, 단일 여성 화자) 사용.
  - 입력은 16,000 샘플 청크를 $[-1, 1]$로 정규화. 멜 스펙트로그램을 지역 조건으로 사용.
  - WaveNet, Gaussian IAF (ClariNet)을 기준 모델로 사용.
  - Adam 옵티마이저($10^{-3}$ 학습률), 200K 반복마다 0.5씩 학습률 감쇠. 배치 크기 8.
  - **FloWaveNet 구조**: 8개의 컨텍스트 블록, 각 블록에 6개의 흐름 (총 48개의 흐름 스택). 각 흐름에 2-레이어 비인과적 WaveNet 아키텍처(커널 크기 3)를 아핀 커플링 레이어의 $m$과 $s$로 사용. 잔차, 스킵, 게이트 채널은 256개.
  - 단일 최대 우도 손실로 700K 반복 동안 훈련. 보조 손실 항 없음.

## 📊 Results

- **음질 및 추론 속도 비교 (Table 1 & 2)**:
  - **MOS (Mean Opinion Score)**: Ground Truth (4.67) > Gaussian WaveNet (4.46) > MoL WaveNet (4.30) > FloWaveNet (3.95) > Gaussian IAF (3.75).
  - **CLL (Conditional Log-Likelihood)**: WaveNet 모델들이 가장 높음. FloWaveNet은 4.5457로 Gaussian IAF보다 높음.
  - **추론 속도**: FloWaveNet은 22,050Hz 오디오 신호를 실시간보다 약 20배 빠르게 생성(420K samples/sec). Parallel WaveNet (500K samples/sec)과 유사한 수준. 기존 WaveNet은 172 samples/sec로 매우 느림.
  - **훈련 속도**: FloWaveNet은 단일 GPU에서 0.71 iter/sec. 2단계 훈련인 Gaussian IAF (0.63 iter/sec)는 총 17.8 GPU-days가 걸리는 반면, FloWaveNet은 11.3 GPU-days로 더 빠른 수렴을 보여줌.
- **FloWaveNet 오디오 품질**: 재현된 Gaussian IAF에서 들리는 백색 잡음이 없었으며, 더 깨끗한 사운드 품질을 보임. 하지만 특정 온도(temperature)에서는 주기적인 "떨리는 목소리"와 같은 아티팩트가 관찰됨.
- **ClariNet 손실 항 분석 (Table 3 & Figure 4)**:
  - KL 발산만 사용한 Gaussian IAF는 최적의 KL 발산(0.040)을 보였지만, 생성된 오디오는 음량이 작고 왜곡됨 (모드 붕괴 문제).
  - KL 발산과 스펙트로그램 프레임 손실을 함께 사용했을 때, 모델이 고조파를 적절히 추적하고 원래 진폭을 추정하여 현실적인 음성 합성이 가능함 (0.134 KL 발산).
  - 스펙트로그램 프레임 손실만 사용한 모델은 빠른 음향 콘텐츠 추정을 보였지만, 많은 양의 잡음이 줄어들지 않음 (1.378 KL 발산).
  - 결론적으로, ClariNet 계열 모델은 현실적인 음성 합성을 위해 보조 손실 항이 필수적임을 입증.
- **WaveNet 확장 컨볼루션의 인과성 (Table 4)**:
  - FloWaveNet에서 비인과적(non-causal) WaveNet 아키텍처를 사용했을 때 MOS 3.95로, 인과적(causal) WaveNet (MOS 3.36)보다 더 나은 음질을 보였습니다. 이는 비인과적 버전이 멜 스펙트로그램 조건을 수용장에서 앞뒤 모두 관찰할 수 있는 이점 때문입니다.

## 🧠 Insights & Discussion

- **FloWaveNet의 장점**: Flow-based 모델인 FloWaveNet은 단일 최대 우도 손실과 단일 단계 훈련만으로 이전 2단계 접근 방식에 필적하는 실시간 병렬 오디오 합성을 달성했습니다. 이는 복잡한 보조 손실 항의 필요성을 줄이고 훈련의 안정성을 유지하여 실용적인 응용 분야에 유용합니다.
- **온도(Temperature)의 영향**: 흐름 기반 생성 모델에서 샘플링 시 온도를 조절하는 것은 생성된 오디오의 지각적 품질에 영향을 미칩니다. 온도를 낮추면(예: 0.8) "떨리는 목소리"를 최소화하여 고품질 음성을 생성할 수 있지만, 지나치게 낮추면 여러 피치에서 일정한 노이즈가 발생할 수 있습니다. 온도를 높이면 노이즈는 없지만, 고조파 구조를 포착하기 어려워지는 "떨리는 목소리"가 나타날 수 있습니다. 고품질 충실도와 음성의 고조파 구조 사이에는 상충 관계가 존재하며, 최적의 지각적 오디오 품질을 위해 경험적 사후 처리(온도 최적화)가 필요합니다.
- **이전 병렬 모델의 한계**: Gaussian IAF (ClariNet)에 대한 분석은 좋은 KL 발산 결과가 반드시 현실적인 오디오를 의미하지 않음을 보여줍니다. 확률 밀도 증류 손실만으로는 모드 붕괴 문제가 발생하기 쉬우며, 현실적인 음성 합성을 위해서는 스펙트로그램 프레임 손실과 같은 보조 손실 항이 필수적입니다.
- **비인과성(Non-causality)의 이점**: FloWaveNet의 흐름 기반 변환은 본질적으로 병렬적이기 때문에 WaveNet 컨볼루션의 인과성(causality)이 더 이상 요구 사항이 아닙니다. 비인과적 WaveNet 구조를 사용하면 멜 스펙트로그램 조건을 양방향으로 관찰할 수 있어 더 나은 음질을 얻을 수 있습니다.

## 📌 TL;DR

FloWaveNet은 느린 WaveNet의 한계와 복잡한 2단계 훈련을 요구하는 기존 병렬 모델의 문제를 해결하기 위해 제안된 흐름 기반(flow-based) 생성 모델입니다. 이 모델은 단일 최대 우도 손실과 단일 훈련 단계만으로 실시간 병렬 오디오 합성을 가능하게 하며, 기존 2단계 접근 방식과 유사한 높은 오디오 충실도를 달성합니다. FloWaveNet은 멜 스펙트로그램 조건이 주어졌을 때 비인과적 WaveNet 아키텍처를 포함하는 아핀 커플링 레이어와 스퀴즈 연산을 활용하여 데이터를 변환합니다. 실험 결과, FloWaveNet은 기존 모델 대비 경쟁력 있는 MOS 점수와 20배 빠른 실시간 추론 속도를 보였으며, 2단계 훈련 방식보다 효율적인 훈련 시간을 가집니다. 또한, 보조 손실 없이도 안정적인 훈련이 가능함을 입증하며, 흐름 기반 모델이 음성 합성 분야에서 유망한 방향임을 제시합니다.
