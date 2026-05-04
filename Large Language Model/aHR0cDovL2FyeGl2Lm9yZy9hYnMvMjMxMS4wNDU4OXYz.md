# TEAL: Tokenize and Embed All for Multi-Modal Large Language Models

Zhen Yang, Yingxue Zhang, Fandong Meng, Jie Zhou (2024)

## 🧩 Problem to Solve

현재의 Multi-modal Large Language Models(MM-LLMs)는 텍스트 외의 다양한 모달리티를 처리함에 있어 비효율적인 구조적 한계를 가지고 있다. 기존의 접근 방식은 크게 두 가지로 나뉘는데, 하나는 처음부터 멀티모달 데이터를 학습시키는 방식이고, 다른 하나는 사전 학습된 텍스트 LLM을 백본으로 사용하고 어댑터(Adapter)를 통해 다른 모달리티의 Dense Feature를 정렬하는 방식이다.

후자의 방식은 텍스트는 토큰 시퀀스로 처리하는 반면, 이미지나 오디오는 Dense Feature(고차원 벡터) 형태로 인코딩하여 입력한다. 이러한 **비통합적 처리(Non-unified processing)**는 LLM이 서로 다른 모달리티 간의 상호작용을 모델링하는 데 부담을 주며, 특히 비텍스트 콘텐츠를 생성하는 능력(Generation)을 현저히 떨어뜨린다. 이를 해결하기 위해 외부 생성 도구(예: Stable Diffusion)를 연결하는 방식이 제안되었으나, 이는 정보 손실과 시스템 복잡도 증가라는 문제를 야기한다.

따라서 본 논문의 목표는 모든 모달리티의 입력을 토큰 시퀀스로 처리하고, 이를 하나의 통합 임베딩 공간(Joint Embedding Space)에서 학습시켜 frozen LLM이 이해(Understanding)와 생성(Generation) 작업을 모두 효율적으로 수행하게 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"Token-in-Token-out"** 패러다임이다. 즉, 이미지와 오디오를 포함한 모든 입력 모달리티를 텍스트와 동일하게 이산적인 토큰 시퀀스로 변환하여 LLM에 입력하고, LLM이 예측한 토큰 시퀀스를 다시 해당 모달리티로 복원하는 구조이다.

주요 기여 사항은 다음과 같다:
1. **TEAL(Tokenize and Embed All) 프레임워크 제안**: 모든 모달리티를 토큰 시퀀스로 취급하고 공동 임베딩 공간을 학습함으로써, frozen LLM이 최소한의 튜닝만으로 멀티모달 이해 및 생성 작업을 수행할 수 있게 한다.
2. **이미지 및 오디오 모달리티의 통합 구현**: frozen LLM을 활용하여 이미지와 오디오라는 두 가지 비텍스트 모달리티를 동시에 처리하는 시스템을 성공적으로 구현하였다.
3. **토크나이저의 중요성 입증**: 다양한 이미지/오디오 토크나이저 실험을 통해, 시맨틱 정보가 풍부한 토크나이저를 설계하는 것이 MM-LLM의 성능 향상에 결정적인 역할을 함을 밝혀냈다.

## 📎 Related Works

### MM-LLMs
기존 연구들은 주로 사전 학습된 비전/오디오 인코더를 사용하여 특징을 추출하고, 이를 LLM의 특징 공간에 정렬시키는 방식(예: BLIP-2, Flamingo, LLaVA)을 사용하였다. 하지만 이러한 방식은 텍스트와 비텍스트 입력의 처리 방식이 달라 상호작용 모델링이 어렵다는 한계가 있다.

### Non-textual Discretization (비텍스트 이산화)
연속적인 이미지나 오디오 신호를 이산적인 토큰 시퀀스로 변환하는 연구들이 진행되어 왔다.
- **VQ-VAEs**: Vector Quantization을 통해 이미지를 이산 코드로 변환하는 VQGAN, BEiT-V2 등이 대표적이다. 오디오 분야에서는 SoundStream 등이 RVQ(Residual Vector Quantizer)를 사용하여 신호를 압축한다.
- **Clustering**: HuBERT나 Whisper 같은 자기지도학습 모델의 특징값에 K-means clustering을 적용하여 클러스터 인덱스를 토큰으로 사용하는 방식이다. 이는 모델 구조 변경 없이 적용 가능하다는 유연성이 있다.

TEAL은 이러한 이산화 기술을 활용하여 모든 모달리티를 토큰화함으로써 LLM과의 호환성을 극대화하였다.

## 🛠️ Methodology

### 전체 시스템 구조
TEAL의 전체 파이프라인은 **입력 $\rightarrow$ 토크나이저 $\rightarrow$ 통합 임베딩 공간 $\rightarrow$ Frozen LLM $\rightarrow$ 디토크나이저 $\rightarrow$ 출력**의 흐름을 가진다.

1. **Tokenization**: 입력된 모달리티(텍스트, 이미지, 오디오)를 각각의 전용 토크나이저를 통해 토큰 시퀀스로 변환한다.
2. **Embedding & Projection**: 비텍스트 토큰을 위한 별도의 임베딩 행렬(Non-textual embedding matrix)과 출력 행렬(Output matrix)을 도입한다. 이때, 텍스트 임베딩 공간과의 차원을 맞추고 의미적 정렬을 위해 **Projection Layer**를 배치한다.
3. **LLM Processing**: LLaMA와 같은 Frozen LLM은 입력된 통합 토큰 시퀀스를 처리하여 다음 토큰을 autoregressive하게 예측한다.
4. **De-tokenization**: 예측된 토큰 시퀀스를 해당 모달리티의 디토크나이저(Decoder)에 통과시켜 최종 결과물(이미지, 오디오 등)을 생성한다.

### 사용된 토크나이저 및 디토크나이저
- **이미지**: DALL-E, VQ-GAN, BEiT-V2를 테스트하였으며, 생성 작업에는 VQGAN을 사용하였다.
- **오디오**: Whisper-small 모델의 11번째 레이어 특징값에 K-means clustering을 적용하여 이산 토큰을 생성하였다.

### 학습 절차 (Two-stage Supervised Fine-tuning)
모델은 파라미터 업데이트를 최소화하기 위해 두 단계로 학습된다.

**1단계: Pre-training (Alignment)**
- **목표**: 비텍스트 임베딩 공간을 텍스트 임베딩 공간과 정렬하는 것이다.
- **방법**: LLM의 모든 파라미터를 고정(Freeze)하고, 오직 두 개의 Projection Layer만 학습시킨다.
- **데이터 포맷**: `[img][text]` 또는 `[text][img]` 형태의 단순한 쌍을 사용한다.

**2단계: Fine-tuning (Downstream Tasks)**
- **목표**: 실제 하위 작업(Understanding & Generation)에 최적화하는 것이다.
- **방법**: LLM 본체는 여전히 고정하고, 비텍스트 관련 파라미터(Embedding, Output Matrix, Projection)를 학습시킨다.
- **특이사항**: 성능 향상을 위해 각 레이어에 Bias와 Norm 파라미터를 추가하는 **Bias-norm tuning**을 적용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: COCO Caption (이미지-텍스트), ScienceQA (멀티모달 QA), CoVoST 2 (오디오 ASR), MNIST (이미지 생성).
- **지표**: CiDER, BLEU-4, Accuracy, Word Error Rate (WER).
- **비교 대상**: LLaMA-Adapter v2, BLIP-2, LLaVA 등.

### 주요 결과
1. **이미지 이해 (COCO Caption & ScienceQA)**:
   - COCO Caption에서 LLaMA-Adapter v2 대비 BLEU-4는 6.6점, CiDER는 1.9점 향상되었다.
   - ScienceQA에서는 LLaMA-Adapter 대비 평균 약 2%p의 성능 향상을 보였다.
2. **오디오 이해 (CoVoST 2 ASR)**:
   - Dense Feature를 사용한 LLaMA-Adapter 대비 WER(Word Error Rate)을 2.74 낮추어, 토큰 기반 방식이 오디오 이해에 더 효율적임을 입증하였다.
3. **이미지 생성 (MNIST)**:
   - 단순한 숫자 생성뿐만 아니라 "3 더하기 8의 마지막 자리 숫자"와 같은 추론이 필요한 프롬프트에 대해서도 올바른 이미지를 생성하였다. 이는 LLM의 텍스트 추론 능력이 유지되면서 생성 능력이 결합되었음을 보여준다.

## 🧠 Insights & Discussion

### 토크나이저의 영향력
실험 결과, 어떤 토크나이저를 사용하느냐가 최종 성능에 결정적인 영향을 미쳤다. 특히 이미지의 경우 BEiT-V2가 VQ-GAN보다 월등히 높은 성능을 보였는데, 이는 BEiT-V2가 사전 학습 과정에서 풍부한 **시맨틱 정보(Semantic information)**를 학습했기 때문으로 분석된다. 즉, 단순한 픽셀 복원력보다 의미론적 정보가 포함된 토큰화가 LLM과의 정렬에 더 중요하다.

### 오디오 어휘집(Vocab Size) 분석
오디오 clustering 시 Vocab size를 1024에서 8192로 늘렸을 때 WER이 18%p 이상 개선되었다. 이는 clustering 기반 방식이 VQ-based 방식보다 어휘집 크기를 유연하게 조정할 수 있다는 강점이 있음을 시사한다.

### 모델 구성 요소의 중요성
Ablation study를 통해 1단계 Pre-training과 Bias-norm tuning이 성능에 매우 중요한 역할을 함을 확인하였다. 반면, 임베딩 초기화 방식(Random vs Tokenizer Codebook)은 데이터 양이 충분하다면 성능 차이가 크지 않을 것으로 추측된다.

## 📌 TL;DR

본 논문은 이미지, 오디오, 텍스트를 모두 동일한 토큰 시퀀스로 취급하여 처리하는 **TEAL(Token-in-Token-out)** 프레임워크를 제안하였다. 모든 모달리티를 통합 임베딩 공간에 정렬함으로써, frozen LLM을 그대로 유지하면서도 효율적인 멀티모달 이해와 생성이 가능함을 입증하였다. 특히 시맨틱 정보가 풍부한 토크나이저의 중요성을 강조하였으며, 이 접근 방식은 향후 단일 모델 내에서 모든 AI 작업을 수행하는 일반 인공지능(AGI) 구현을 위한 유망한 방향성을 제시한다.