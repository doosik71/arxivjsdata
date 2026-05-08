# IMPROVING HYBRID CTC/ATTENTION END-TO-END SPEECH RECOGNITION WITH PRETRAINED ACOUSTIC AND LANGUAGE MODELS

Keqi Deng, Songjun Cao, Yike Zhang, Long Ma (2021)

## 🧩 Problem to Solve

본 논문은 종단간(End-to-End, E2E) 자동 음성 인식(Automatic Speech Recognition, ASR) 모델, 특히 Sequence-to-Sequence(S2S) 구조에서 사전 학습된 음향 모델(Acoustic Model, AM)과 언어 모델(Language Model, LM)을 효율적으로 활용하지 못하는 문제를 해결하고자 한다.

전통적인 S2S E2E 모델의 디코더는 인코더의 음향 표현(acoustic representation)에 강하게 의존하도록 설계되어 있어, 디코더만 별도로 사전 학습시키는 것이 어렵다. 또한, BERT나 GPT2와 같은 기존의 강력한 사전 학습 언어 모델들을 ASR 디코더에 적용하려 해도 아키텍처의 불일치(architecture mismatch)로 인해 파라미터 초기화에 활용하기 어렵다는 한계가 있다. 레이블이 지정된 데이터의 부족은 E2E ASR 모델이 산업 현장에 배포된 성숙한 모델들에 비해 성능이 떨어지는 주요 원인이 되므로, 대규모 비정형 데이터를 통해 학습된 사전 학습 모델을 S2S ASR 시스템에 통합하는 것은 매우 중요한 과제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 사전 학습된 AM과 LM을 모두 활용할 수 있는 **Preformer**라는 hybrid CTC/attention S2S ASR 아키텍처를 제안한 것이다.

가장 중점적인 설계 아이디어는 **One-Cross Decoder (OCD)**의 도입이다. 기존 Transformer 디코더는 모든 레이어에서 교차 주의 집중(cross-attention)을 수행하여 인코더 출력에 과도하게 의존하지만, OCD는 단 한 번의 cross-attention만을 수행하도록 설계되었다. 이를 통해 디코더의 대부분을 인코더 출력과 독립적인 구조로 만들어, 사전 학습된 언어 모델인 DistilGPT2로 초기화할 수 있게 하였다. 또한, 인코더는 사전 학습된 wav2vec 2.0을 사용하고, CTC(Connectionist Temporal Classification)를 보조 작업으로 활용하여 학습의 수렴 속도와 긴 발화 인식 능력을 향상시켰다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 언급한다.

1. **음향 모델 사전 학습**: wav2vec 2.0과 같은 자기지도 학습(self-supervised learning) 기반 모델들이 뛰어난 성능을 보였으나, attention 기반의 S2S ASR 모델에 적용했을 때 그 효과가 제한적이었다는 점을 지적한다.
2. **언어 모델 사전 학습**: BERT나 GPT2 같은 모델들이 NLP 분야에서 성공을 거두었으며, ASR에서는 이를 위해 shallow fusion, cold fusion 또는 지식 증류(knowledge distillation) 방식이 사용되었다. 그러나 이러한 방식들은 S2S ASR 모델의 내부 구조에 사전 학습된 LM을 직접적으로 통합하여 파라미터를 초기화하는 방식이 아니었기에 효율성이 떨어졌다고 분석한다.

본 연구는 단순한 외부 LM 결합이나 지식 증류를 넘어, 모델 아키텍처 자체를 수정하여 사전 학습된 AM과 LM을 S2S 구조의 인코더와 디코더에 직접 통합했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

Preformer는 크게 **w2v-encoder**, **One-Cross Decoder (OCD)**, 그리고 CTC 작업을 위한 **Fully Connected (FC) layer**로 구성된다.

### 주요 구성 요소 및 역할

1. **w2v-encoder**: 사전 학습된 wav2vec 2.0을 기반으로 하며, CNN 기반 특성 추출기와 Transformer 기반 컨텍스트 네트워크로 이루어져 있다. 원시 음성 신호를 음향 표현으로 변환한다. 파인튜닝 시에는 Transformer 컨텍스트 네트워크의 파라미터만 업데이트한다.
2. **One-Cross Decoder (OCD)**: 디코더의 음향 표현 의존성을 완화한 구조이다.
    - **Self Layers**: Multi-head self-attention과 feed-forward 모듈로 구성된 레이어들이다. 이 부분은 사전 학습된 DistilGPT2로 초기화되며, 인코더의 출력 없이 텍스트 문맥 정보만을 처리한다.
    - **Cross Layer**: 마지막에 단 한 번 배치되는 레이어로, Multi-head cross-attention과 feed-forward 모듈로 구성된다. 여기서 비로소 인코더의 음향 표현이 입력된다.
3. **CTC Branch**: w2v-encoder 뒤에 FC 레이어를 추가하여 CTC 손실을 계산한다. 이는 인코더의 수렴을 돕고 추론 시 시간 경계(time boundaries)를 정교하게 잡는 역할을 한다.

### 학습 목표 및 손실 함수

학습 시에는 CTC 손실과 교차 엔트로피(Cross-Entropy) 손실을 함께 사용하는 다중 작업 학습(Multi-task learning) 목적 함수를 사용한다.

$$L_{mtl} = \lambda L_{ctc} + (1-\lambda) L_{ce}$$

여기서 $\lambda$는 CTC 분기의 가중치를 나타낸다.

### 추론 절차 및 디코딩

추론 단계에서는 CTC 스코어와 OCD 스코어를 결합하여 최종 디코딩 점수를 산출한다.

$$S = \mu S_{ctc} + (1-\mu) S_{ocd}$$

여기서 $\mu$는 CTC 분기 점수의 가중치를 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 중국어 AISHELL-1 코퍼스 (178시간). AM 사전 학습을 위해 AISHELL-2의 비정형 데이터를 사용하였다.
- **기준선(Baseline)**: vanilla hybrid CTC/attention Transformer 모델.
- **측정 지표**: 문자 오류율(Character Error Rate, CER).

### 주요 결과

실험 결과, 제안된 Preformer는 테스트 셋에서 **4.6% CER**을 달성하였다. 이는 vanilla Transformer baseline(6.3% CER) 대비 **상대적으로 27%의 CER 감소**를 가져온 수치이다.

| ASR Model | Dev CER (%) | Test CER (%) |
| :--- | :---: | :---: |
| Our Transformer baseline | 5.9 | 6.3 |
| **Our Preformer** | **4.3** | **4.6** |
| ESPnet2 (Conformer+SpecAug) | 4.4 | 4.7 |

특히, Preformer는 데이터 증강 기법인 SpecAugment를 사용하지 않고도, 이를 적용한 Conformer 모델의 성능을 상회하는 결과를 보여주었다.

### 절제 연구 (Ablation Studies)

1. **OCD 분석**: DistilGPT2로 초기화된 파라미터를 고정(fixed)했을 때 성능이 가장 좋았다. 이는 제한된 레이블 데이터로 파인튜닝할 경우 오히려 과적합(overfitting)이 발생하여 사전 학습된 LM의 고품질 텍스트 표현 능력이 훼손될 수 있음을 시사한다.
2. **Cross-attention 횟수**: cross-attention 레이어를 두 개 사용한 TCD(Two-Cross Decoder)가 OCD보다 성능이 좋지 않았다. 이는 고품질의 텍스트 표현이 있다면 단 한 번의 cross-attention만으로도 충분하다는 가설을 뒷받침한다.
3. **w2v-encoder 분석**: 영어 wav2vec 2.0 모델로 초기화하더라도 vanilla Transformer보다 성능이 좋았으며, 중국어 전용 모델을 사용했을 때 최적의 성능을 보였다. 이는 교차 언어 사전 학습이 저자원 언어 ASR에 도움이 될 수 있음을 의미한다.

## 🧠 Insights & Discussion

본 논문은 S2S ASR 모델의 고질적인 문제였던 "디코더의 인코더 의존성"을 아키텍처 수정(OCD)을 통해 해결함으로써, 사전 학습된 AM과 LM을 동시에 통합하는 데 성공하였다.

**강점 및 통찰**:

- **구조적 분리**: 디코더에서 cross-attention을 마지막 단계로 밀어냄으로써 LM의 사전 학습 지식을 그대로 가져올 수 있는 구조적 통로를 마련하였다.
- **상호 보완적 학습**: CTC 보조 손실을 통해 인코더의 학습을 안정화하고, 고품질의 LM 표현을 통해 디코더의 예측력을 높이는 상호 보완적 전략이 유효함을 증명하였다.

**한계 및 논의사항**:

- **파라미터 고정**: LM 파라미터를 고정하는 것이 최적의 성능을 냈다는 점은, 여전히 S2S 모델에서 사전 학습 모델을 미세 조정(fine-tuning)하는 과정에 과적합 위험이 크다는 것을 보여준다.
- **데이터 의존성**: 본 연구는 AISHELL-1이라는 특정 데이터셋에서 검증되었으므로, 더 다양한 언어나 극단적인 저자원 환경에서도 동일한 성능 향상 폭이 나타날지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 사전 학습된 음향 모델(wav2vec 2.0)과 언어 모델(DistilGPT2)을 통합한 **Preformer** 아키텍처를 제안한다. 핵심은 디코더의 의존성을 낮춘 **One-Cross Decoder (OCD)**를 도입하여 사전 학습된 LM을 직접 활용 가능하게 만든 것이며, 이를 통해 AISHELL-1 데이터셋에서 기존 baseline 대비 CER을 27% 상대적으로 감소시키는 성과를 거두었다. 이 연구는 향후 저자원 언어의 ASR 시스템 구축 시 사전 학습 모델을 효율적으로 통합하는 방법론으로 활용될 가능성이 높다.
