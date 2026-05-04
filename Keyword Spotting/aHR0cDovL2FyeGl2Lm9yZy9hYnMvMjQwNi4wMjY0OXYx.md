# Keyword-Guided Adaptation of Automatic Speech Recognition

Aviv Shamsian, Aviv Navon, Neta Glazer, Gill Hetz, Joseph Keshet (2024)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템이 직면한 두 가지 주요 문제인 **소음이 심한 환경**과 **특수 용어(Jargon)** 인식의 어려움을 해결하고자 한다. 최신 ASR 모델들은 대규모 데이터셋을 통해 전반적인 성능을 높였으나, 산업 현장의 기계 소음이나 의료, 법률 분야와 같이 일상 대화에서 드물게 사용되는 전문 용어가 포함된 경우 인식률이 급격히 저하되는 한계가 있다.

따라서 본 연구의 목표는 도메인 특화 지식이나 개인화된 키워드 정보를 활용하는 **Contextual Biasing** 기법을 도입하여, Whisper 기반 모델이 특정 키워드를 더 정확하게 인식하도록 유도함으로써 전체적인 Word Error Rate(WER)를 낮추고 전문 용어에 대한 재현율(Recall)을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Open Vocabulary Keyword Spotting(KWS) 모델을 사용하여 ASR 디코더를 동적으로 가이드하는 프롬프트를 생성**하는 것이다.

구체적인 기여 사항은 다음과 같다.
1. **KWS 기반 동적 프롬프팅**: Whisper의 인코더 표현(representation)을 활용하는 AdaKWS 모델을 통해 입력 음성에서 키워드를 식별하고, 이를 디코더의 프롬프트로 입력하여 전사(transcription) 과정을 안내한다.
2. **두 가지 적응 방법론 제안**:
    - **KG-Whisper**: Whisper의 디코더 파라미터를 직접 미세 조정(Fine-tuning)하는 방식이다.
    - **KG-Whisper-PT**: 전체 모델을 동결하고 학습 가능한 작은 크기의 프롬프트 접두사(Prompt Prefix)만을 학습시키는 Prompt Tuning 방식이다.
3. **강력한 일반화 성능 입증**: 학습 과정에서 보지 못한 새로운 도메인(의료, 항공 관제) 및 새로운 언어에 대해서도 성능 향상이 있음을 실험적으로 증명하였다.

## 📎 Related Works

기존의 Contextual Biasing 접근 방식으로는 Beam Search에 컨텍스트를 주입하거나 Shallow Fusion 및 Deep Fusion 기법을 사용하는 방법들이 있었다. 최근에는 Whisper 모델에 도메인 프롬프트를 사용하여 특정 도메인에 적응시키는 연구가 진행되었으나, 이러한 방법들은 대량의 미세 조정 데이터가 필요하며 제공할 수 있는 사전 지식의 범위에 제한이 있다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 **Open Vocabulary KWS**를 통합함으로써, 특정 단어 리스트를 유연하게 제공하고 이를 통해 디코더를 유도하는 방식을 취한다. 이는 고정된 도메인 프롬프트를 사용하는 대신, 입력 데이터에 따라 동적으로 변화하는 키워드 기반 프롬프트를 사용한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
시스템은 크게 **Whisper Encoder $\rightarrow$ AdaKWS $\rightarrow$ Prompt Generation $\rightarrow$ Whisper Decoder** 순으로 구성된다.

1. **Encoder**: 입력 음성 $x \in X^T$가 들어오면 Whisper 인코더 $f^e_\phi$를 통해 음향 표현 $u \in \mathbb{R}^F$를 생성한다.
2. **KWS**: AdaKWS 모델 $f^k_\theta$는 인코더 표현 $u$와 키워드 집합 $K$를 입력받아 각 키워드의 존재 여부를 나타내는 이진 벡터 $\hat{y} \in \{0, 1\}^{|K|}$를 예측한다.
3. **Prompting**: KWS가 탐지한 키워드들을 결합하여 프롬프트 $p_{KWS}$를 생성하고, 이를 Whisper 디코더의 입력으로 제공한다.

### 학습 목표 및 손실 함수
ASR의 기본 목표는 다음의 Cross-Entropy(CE) 손실 함수를 최소화하는 것이다.
$$\min_{\{\phi, \psi\}} \sum_{j} \sum_{i} L_{CE}(f^d_\psi(f^e_\phi(x_j), p_j, t_{i-1}^j), t_i^j)$$
여기서 $f^d_\psi$는 디코더, $p$는 프롬프트, $t$는 토큰 시퀀스를 의미한다.

본 논문에서 제안하는 두 가지 방법의 최적화 식은 다음과 같다.

**1. KG-Whisper (Decoder Fine-tuning)**
인코더를 동결하고 디코더 파라미터 $\psi$만을 학습시킨다.
$$\min_{\psi} \sum_{j} \sum_{i} L_{CE}(f^d_\psi(u_j, p_{KWS}(u_j, K), t_{i-1}^j), t_i^j)$$

**2. KG-Whisper-PT (Prompt Tuning)**
인코더와 디코더를 모두 동결하고, 학습 가능한 프롬프트 접두사 $q \in \mathbb{R}^{N \times D}$만을 학습시킨다.
$$\min_{q} \sum_{j} \sum_{i} L_{CE}(f^d_\psi(u_j, [q, p_{KWS}(u_j, K)], t_{i-1}^j), t_i^j)$$

### 훈련 전략
학습 시 KWS의 예측을 모사하기 위해 다음과 같은 동적 샘플링 전략을 사용한다.
- 프롬프트 내 키워드 개수는 1~5개 사이에서 무작위로 선택한다.
- 각 키워드가 긍정(Positive)일 확률을 0.9로 설정하여, 실제 정답 텍스트에서 키워드를 샘플링하거나 다른 샘플에서 부정(Negative) 키워드를 샘플링한다.
- 키워드는 $|$ 구분자로 연결되며, $\langle \text{start\_of\_prev} \rangle$와 $\langle \text{start\_of\_transcript} \rangle$ 토큰 사이에 배치된다.

## 📊 Results

### 실험 설정
- **데이터셋**: Voxpopuli(다국어), UWB-ATCC(항공 관제, 고소음), Medical(의료 전문 용어), Fleurs(저자원 언어).
- **평가 지표**: Word Error Rate(WER) 및 긍정 키워드에 대한 F1 Score.
- **기준선(Baselines)**: Pre-trained Whisper, Whisper + prompt(미세 조정 없이 프롬프트만 추가), Whisper FT(일반 디코더 미세 조정), Whisper PT(일반 프롬프트 튜닝).

### 주요 결과
1. **다국어 성능 (Voxpopuli)**:
   - KG-Whisper와 KG-Whisper-PT 모두 기본 Whisper 모델보다 월등한 성능을 보였다.
   - 특히 KG-Whisper-PT는 단 **15K 개의 학습 파라미터**만으로도 WER을 2.3% 감소시켰으며, F1 스코어를 크게 향상시켰다.
   - 단순하게 프롬프트만 추가한 'Whisper + prompt'는 오히려 성능이 하락했는데, 이는 미세 조정 없는 프롬프트가 모델의 환각(Hallucination)을 유발하기 때문으로 분석된다.

2. **도메인 일반화 (Out-of-domain)**:
   - **UWB-ATCC (고소음)**: KG-Whisper-PT가 Whisper 대비 WER과 F1 모두에서 뚜렷한 개선을 보였다.
   - **Medical (전문 용어)**: 의료 분야의 특수 용어 인식률이 크게 향상되었으며, KG-Whisper-PT의 경우 F1 스코어가 96.58%에 달했다.

3. **미학습 언어 일반화 (Unseen Languages)**:
   - Fleurs 데이터셋의 6개 미학습 언어에 대해 테스트한 결과, Whisper 대비 **평균 5.1%의 WER 개선**을 달성하였다.

### Ablation Study (프롬프트 길이)
학습 가능한 토큰 수($N$)에 따른 성능을 분석한 결과, **12~16개 토큰**일 때 최적의 성능을 보였다.
- **너무 짧은 경우**: 디코더에 제공되는 컨텍스트가 부족하여 단어 누락(Deletion error)이 증가한다.
- **너무 긴 경우**: 모델이 불필요한 텍스트를 생성하는 환각 현상이 발생하여 삽입 오류(Insertion error)가 증가하고 연산 비용이 상승한다.

## 🧠 Insights & Discussion

### 강점
본 연구는 거대한 ASR 모델을 전체적으로 재학습시키지 않고도, KWS라는 외부 모듈과 효율적인 Prompt Tuning(KG-Whisper-PT)을 통해 도메인 특화 능력을 부여할 수 있음을 보여주었다. 특히 매우 적은 수의 파라미터 수정만으로도 전문 용어 인식률을 획기적으로 높인 점과, 이를 보지 못한 언어와 소음 환경으로 확장한 일반화 능력이 매우 강력하다는 점이 돋보인다.

### 한계 및 논의사항
- **KWS 의존성**: 시스템의 성능이 AdaKWS의 키워드 탐지 정확도에 의존한다. KWS가 키워드를 잘못 탐지하여 프롬프트로 제공할 경우, 디코더가 잘못된 유도를 받아 오인식할 가능성이 존재한다.
- **환각 현상**: Ablation study에서 언급되었듯, 프롬프트의 길이나 구성이 부적절할 경우 ASR 모델 특유의 환각 현상이 발생할 수 있다. 이는 Whisper와 같은 대형 생성 모델 기반 ASR의 고질적인 문제이며, 이를 완전히 제어하기 위한 추가적인 제약 메커니즘이 필요할 수 있다.

## 📌 TL;DR

본 논문은 Whisper ASR 모델이 전문 용어 및 소음 환경에서 겪는 성능 저하를 해결하기 위해, **Open Vocabulary KWS(AdaKWS)를 통해 탐지된 키워드를 디코더의 프롬프트로 주입하는 방식**을 제안한다. 특히 디코더 전체를 튜닝하는 KG-Whisper와 소량의 접두사만 학습하는 KG-Whisper-PT 두 가지 경로를 제시하였으며, KG-Whisper-PT는 단 15K 개의 파라미터만으로도 전문 용어 인식률을 극대화하고 미학습 언어에 대해서도 평균 5.1%의 WER 개선을 이루었다. 이 연구는 향후 음성 기반 파운데이션 모델을 특정 산업 도메인에 효율적으로 적응시키는 데 중요한 방법론을 제공한다.