# DenseLoRA: Dense Low-Rank Adaptation of Large Language Models

Lin Mu, Xiaoyu Wang, Li Ni, Yang Li, Zhize Wu, Peiquan Jin, Yiwen Zhang (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)을 특정 태스크에 적응시키기 위한 효율적인 파라미터 튜닝 방법론을 다룬다. 기존의 대표적인 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 기법인 LoRA(Low-Rank Adaptation)는 두 개의 저차원 행렬을 통해 가중치 업데이트를 근사함으로써 학습 파라미터 수를 획기적으로 줄였다.

그러나 저자들은 LoRA의 저차원 행렬 내의 많은 가중치들이 학습 과정에서 거의 변화하지 않거나 0에 가까운 값을 가지는 '중복성(Redundancy)' 문제를 가지고 있음을 지적한다. 이는 파라미터 활용 효율이 낮음을 의미하며, 결과적으로 더 적은 파라미터로도 더 높은 성능을 낼 수 있는 가능성이 있음을 시사한다. 따라서 본 논문의 목표는 LoRA의 구조적 한계를 극복하여, 더 적은 수의 파라미터를 사용하면서도 더 높은 성능을 달성할 수 있는 '밀집된(Dense)' 구조의 저차원 적응 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 가중치 행렬 자체를 직접 수정하는 대신, **숨겨진 표현(Hidden Representation)을 정제하고 압축한 뒤 적응(Adaptation)시키는 방식**을 도입하는 것이다.

DenseLoRA는 representation fine-tuning의 개념을 통합하여, 모든 적응 레이어에서 공유되는 하나의 Encoder-Decoder 구조를 통해 표현을 압축 및 복원하고, 그 사이에서 매우 작은 크기의 Dense Low-Rank 행렬을 통해 태스크별 적응을 수행한다. 이를 통해 LoRA에서 나타나는 가중치 중복성을 제거하고 파라미터 효율성을 극대화하였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **LoRA (Low-Rank Adaptation):** 가중치 업데이트 $\Delta W$를 두 개의 저차원 행렬 $A$와 $B$의 곱으로 표현한다. 하지만 실제 학습 시 많은 파라미터가 비활성화되어 효율성이 떨어진다는 한계가 있다.
2. **LoRA 변형 기법들:** AdaLoRA(SVD 기반 가지치기), DoRA(크기와 방향 분리), VeRA(공유 무작위 행렬 사용) 등이 효율성을 높이려 시도했으나, 여전히 전통적인 저차원 적응 프레임워크의 제약 내에 머물러 있다.
3. **Representation Fine-tuning:** 가중치가 아닌 은닉 표현을 직접 수정하는 방식(예: ReFT, RED)으로, 매우 적은 파라미터로 태스크 적응이 가능함을 보였으나, 본 논문처럼 저차원 적응 행렬과 결합된 형태는 아니었다.

### 차별점

DenseLoRA는 단순한 가중치 근사가 아니라, **'표현 압축 $\rightarrow$ 밀집 적응 $\rightarrow$ 표현 복원'**이라는 파이프라인을 구축하였다. 특히 Encoder와 Decoder를 모든 레이어에서 공유함으로써 파라미터 수를 획기적으로 줄이면서도, 레이어별로 고유한 Dense 행렬 $M$을 두어 유연성을 확보했다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

DenseLoRA는 다음과 같은 3단계 프로세스로 구성된다.

1. **Compression (압축):** Encoder가 입력 은닉 표현 $h$를 저차원 표현 $h'$로 압축하고 정제한다.
2. **Adaptation (적응):** 압축된 표현 $h'$에 레이어별로 고유한 Dense Low-Rank 행렬 $M$을 적용하여 태스크에 맞게 변환한다.
3. **Reconstruction (복원):** Decoder가 변환된 표현을 원래의 차원으로 복원하여, 프리트레이닝된 모델의 출력과 결합한다.

### 주요 방정식 및 절차

전체 적응 과정은 다음과 같은 수식으로 표현된다.

$$\hat{h} = W_0 h + \text{Decoder}(M \text{Encoder}(h))$$

여기서 $W_0$는 고정된(frozen) 프리트레이닝 가중치 행렬이며, $\hat{h}$는 최종 적응된 표현이다. 각 구성 요소의 세부 동작은 다음과 같다.

* **Encoder:** 입력 $h \in \mathbb{R}^k$를 받아 $\sigma(W_e h)$를 통해 $h' \in \mathbb{R}^r$를 생성한다. 여기서 $W_e \in \mathbb{R}^{r \times k}$이며 $\sigma$는 활성화 함수이다.
* **Adaptation Matrix $M$:** 압축된 표현에 $M \in \mathbb{R}^{r \times r}$ 행렬을 곱한다. $M$은 각 레이어마다 고유하게 존재하며, LoRA의 $A, B$와 달리 매우 작지만 밀집된(dense) 업데이트를 수행한다.
* **Decoder:** 적응된 표현을 $\sigma(W_d^T h')$를 통해 원래 차원 $d$로 복원한다. 여기서 $W_d \in \mathbb{R}^{d \times r}$이다.

### 학습 및 초기화 전략

* **Shared Matrices:** Encoder의 $W_e$와 Decoder의 $W_d$는 모든 레이어에서 공유된다. $W_e$는 Kaiming 초기화를 사용하고, $W_d$는 초기값을 0으로 설정하여 학습 초기 단계에서 모델 출력에 영향을 주지 않도록 한다.
* **Unique Matrices:** 각 레이어의 $M$ 행렬은 Kaiming 초기화를 사용하여 독립적으로 학습된다.

### 파라미터 수 분석

학습 가능한 파라미터 수 $|\Theta|$를 비교하면 다음과 같다 (단, $l$은 레이어 수, $d$는 출력 차원, $k$는 입력 차원, $r$은 랭크).

* **LoRA:** $|\Theta| = l \times (d + k) \times r$
* **DenseLoRA:** $|\Theta| = (d + k + l \times r) \times r$

LoRA는 모든 레이어마다 $d+k$ 크기의 행렬을 가져야 하지만, DenseLoRA는 $d+k$ 부분을 공유함으로써 파라미터 수를 획기적으로 줄인다. 예를 들어 LLaMA2-7B ($r=16$) 기준, LoRA가 28M개의 파라미터를 사용할 때 DenseLoRA는 0.9M개만 사용하여 약 30배의 감소 효과를 보인다.

## 📊 Results

### 실험 설정

* **모델:** LLaMA2-7B, LLaMA3-8B
* **데이터셋:** 상식 추론(Commonsense Reasoning) 8개 벤치마크 및 산술 추론(Arithmetic Reasoning) 4개 벤치마크.
* **비교 대상:** LoRA, LoKr, NoRA, VeRA, AdaLoRA, DoRA 및 ChatGPT (Zero-shot CoT).
* **지표:** Accuracy (%)

### 주요 결과

1. **상식 추론 성능:** LLaMA3-8B 모델에서 DenseLoRA는 학습 파라미터의 단 **0.01%만 사용하고도 83.8%의 정확도**를 기록하였다. 이는 LoRA가 0.70%의 파라미터를 사용하고 80.8%의 정확도를 낸 것에 비해, 파라미터는 70배 적게 쓰면서 성능은 3%p 더 높음을 의미한다.
2. **산술 추론 성능:** LLaMA3-8B 기준, $r=64$ 설정에서 DenseLoRA는 0.06%의 파라미터만으로 58.5%의 정확도를 달성하여 LoRA(0.70% 파라미터, 56.9% 정확도)를 능가하였다.
3. **데이터 효율성:** 학습 데이터의 10%~80%만 샘플링하여 실험한 결과, DenseLoRA는 모든 데이터 규모에서 LoRA보다 우수한 일반화 성능을 보였다. 특히 10%의 데이터만으로 학습한 DenseLoRA가 전체 데이터를 사용한 LoRA보다 높은 정확도를 기록하기도 했다.
4. **튜닝 범위(Granularity) 분석:** Multi-head Attention(QKV)보다 MLP 레이어(Up/Down)를 적응시키는 것이 성능 향상에 더 효과적임을 확인하였다.

## 🧠 Insights & Discussion

### 성능 향상의 원인 분석

저자들은 LoRA의 $A, B$ 행렬과 DenseLoRA의 $M$ 행렬의 업데이트 양상을 비교 분석하였다. LoRA의 행렬들은 업데이트 값이 0에 가까운 부분이 많은 희소(Sparse)한 패턴을 보인 반면, **DenseLoRA의 $M$ 행렬은 대부분의 파라미터가 활발하게 업데이트되는 밀집(Dense)한 패턴**을 보였다. 이는 DenseLoRA가 파라미터를 훨씬 효율적으로 활용하고 있음을 증명한다.

### 구성 요소의 중요성

Encoder와 Decoder에서 활성화 함수($\sigma$)를 제거한 실험(Only Matrix variant)에서 성능 하락이 관찰되었다. 이는 단순한 선형 변환보다 비선형 정제 과정이 표현 효율성을 높이는 데 필수적임을 시사한다.

### 한계 및 비판적 해석

본 논문은 텍스트 기반의 추론 태스크에서 탁월한 효율성을 입증하였으나, 이미지 생성이나 비주얼 인스트럭션 튜닝과 같은 멀티모달 태스크로의 확장성은 아직 검증되지 않았다. 또한, 공유된 Encoder-Decoder가 모든 레이어의 특징을 충분히 포괄할 수 있는지에 대한 이론적 근거보다는 실험적 결과에 의존하고 있다는 점이 논의될 필요가 있다.

## 📌 TL;DR

DenseLoRA는 LoRA의 가중치 중복성 문제를 해결하기 위해 **공유된 Encoder-Decoder 구조와 레이어별 밀집 적응 행렬($M$)**을 결합한 새로운 PEFT 방법론이다. LLaMA3-8B 실험 결과, LoRA 대비 **학습 파라미터를 약 70배 적게 사용하면서도 더 높은 정확도를 달성**하였다. 이 연구는 LLM의 효율적인 적응을 위해 가중치 수정보다 은닉 표현의 정제가 더 효과적일 수 있음을 보여주며, 향후 초소형 파라미터 튜닝 연구에 중요한 방향성을 제시한다.
