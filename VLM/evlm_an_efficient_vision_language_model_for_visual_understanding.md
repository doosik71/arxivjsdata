# EVLM: An Efficient Vision-Language Model for Visual Understanding

Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, Changyi Liu, Dewen Fan, Huihui Xiao, Jiahong Wu, Fan Yang, Size Li, Di Zhang (2024)

## 🧩 Problem to Solve

본 논문은 최근 대규모 멀티모달 언어 모델(MLLM)들이 채택하고 있는 LLaVA 스타일 아키텍처의 한계를 해결하고자 한다. LLaVA 계열의 모델들은 일반적으로 Vision Transformer(ViT)의 단일 레이어 특징(feature)을 시각적 프롬프트로 사용하여 텍스트 토큰과 함께 언어 모델(LLM)에 직접 입력하는 방식을 취한다.

이러한 방식은 다음과 같은 두 가지 주요 문제를 야기한다. 첫째, 고해상도 이미지나 비디오와 같이 시각적 신호의 시퀀스가 길어질 경우, LLM의 Self-attention 메커니즘으로 인해 연산 비용이 기하급수적으로 증가하는 computational overhead 문제가 발생한다. 둘째, ViT의 단일 레이어 특징만을 사용하면 LLM이 시각적 신호를 충분히 포괄적으로 인지하기 어려워 세밀한 이해 능력이 떨어진다는 점이다.

따라서 본 연구의 목표는 연산 비용을 최소화하면서도 시각적 신호를 최대한 종합적으로 인지할 수 있는 효율적인 멀티모달 언어 모델인 EVLM을 설계하는 것이다.

## ✨ Key Contributions

EVLM의 핵심 설계 아이디어는 효율성과 인지 능력의 균형을 맞추는 것이며, 이를 위해 다음 세 가지 핵심 요소를 도입하였다.

1. **Cross-Attention 기반의 상호작용**: Flamingo와 유사하게 이미지와 텍스트 간의 상호작용에 Cross-attention 메커니즘을 사용하여, 시각적 토큰의 길이가 길어지더라도 LLM의 연산 부하를 효과적으로 제어한다.
2. **Hierarchical ViT Features 활용**: ViT의 서로 다른 레이어에서 계층적 특징을 추출하여 LLM에 전달함으로써, 모델이 다양한 입도(granularity)의 시각적 신호를 인식할 수 있도록 한다.
3. **Mixture of Experts (MoE) 도입**: Cross-Attention 레이어에 MoE 메커니즘을 적용하여, 모델의 전체 파라미터 규모를 확장하면서도 추론 시의 효율성을 유지하고 모델의 효과성을 증대시킨다.

## 📎 Related Works

본 논문에서는 MLLM의 입력 투영(Input Project) 방식과 시각 인코더, MoE 구조에 대해 논의한다.

- **입력 투영 방식**: LLaVA와 같은 모델들은 MLP를 통해 시각 특징을 LLM의 입력 공간으로 매핑하여 단순 결합(concatenation)하는 방식을 사용한다. 반면, BLIP-2와 같은 모델은 Q-former를 통해 고정된 수의 토큰으로 시각 정보를 압축한다. 하지만 단순 결합은 연산 비용이 크고, 압축 방식은 정보 손실의 위험이 있다. EVLM은 Flamingo처럼 LLM의 각 레이어에 시각 특징을 깊게 융합하는 방식을 채택하여 이 문제를 해결한다.
- **시각 인코더**: CLIP, EvaCLIP, SigLIP 등이 널리 사용되며, 일부 연구는 DINOv2와 같은 추가 인코더를 사용하여 표현력을 높이거나 경량 CNN을 결합하여 효율성을 높이려 한다.
- **MoE**: MoE는 희소 활성화(sparse activation)를 통해 동일한 연산 자원으로 모델 규모를 확장할 수 있는 기술이다. 최근 DeepSeek-MoE와 같이 세분화된(fine-grained) 전문가 구조를 통해 전문성을 높이는 연구가 진행되고 있으며, EVLM 역시 이를 Gated Cross-Attention 레이어에 적용하였다.

## 🛠️ Methodology

### 전체 시스템 구조

EVLM은 시각 인코더(Visual Encoder), 대규모 언어 모델(LLM), 그리고 Gated Cross-Attention 레이어로 구성된다. 시각 인코더에서 추출된 계층적 특징이 Gated Cross-Attention을 통해 LLM의 각 트랜스포머 레이어 사이에 삽입되어 시각 정보와 텍스트 정보가 융합된다.

### 주요 구성 요소 및 역할

1. **Visual Encoder**: 4.4B 규모의 EVA2-CLIP-E-Plus 모델을 사용한다. 마지막 40개 레이어에서 8개의 특징 시퀀스를 균일하게 샘플링하여 계층적 시각 특징을 추출하며, 이를 각각 서로 다른 Gated Cross-Attention 레이어에 입력한다.
2. **Gated Cross-Attention Layer**: Flamingo의 구조를 따르되, `<image>` 토큰을 16개의 학습 가능한(learnable) 토큰 세트로 대체하여 Q-former와 유사하게 시각 특징을 효율적으로 운반하게 한다. 텍스트 시퀀스가 시각 정보와 무관할 경우를 대비해 all-zero 벡터를 패딩하며, 특정 어텐션 마스크를 통해 토큰 간 상호작용을 제어한다.
3. **LLM**: Qwen-14B-Chat 1.0을 기반으로 하며, 모든 트랜스포머 레이어 앞에 Gated Cross-Attention 레이어를 삽입하여 시각적 조건화를 수행한다.

### 연산 효율성 분석 (FLOPs)

논문은 단순 결합 방식(Full-attention)과 Cross-attention 방식의 연산량을 수식으로 비교하여 효율성을 증명한다.

- **Full-attention FLOPs**:
$$FLOPs_{full-attention} = 24B(s_{img} + s_{txt})h_{llm}^2 + 4B(s_{img} + s_{txt})^2h_{llm}$$

- **Cross-attention FLOPs**:
$$FLOPs_{cross-attention} = 4(6 + r_{xc} + r_{xf})B(16 + s_{txt})h_{llm}^2 + 4B(16 + s_{txt})^2h_{llm} + 4r_{xc}Bs_{img}d_{img}h_{llm} + 4r_{xc}B(16 + s_{txt})s_{img}h_{llm}$$

여기서 $B$는 배치 크기, $s_{img}$와 $s_{txt}$는 각각 시각 및 텍스트 토큰의 길이, $h_{llm}$은 LLM의 은닉 상태 크기, $d_{img}$는 시각 표현의 차원, $r_{xc}$와 $r_{xf}$는 각각 Cross-Attention 및 FFN 레이어의 비율을 의미한다. 실험 결과, 시각 토큰 길이가 길어질수록(예: $s_{img}=1024$) Cross-attention 방식이 연산량을 획기적으로 줄임(S=0.077)을 확인하였다.

### Mixture-of-Experts (MoE) 확장

성능 향상을 위해 Gated Cross-Attention 레이어의 FFN을 세분화된 MoE 구조로 확장한다.

- **구조**: EVLM-Base의 FFN 파라미터를 $N$번 복제하고, 각 복제본을 다시 $M$개의 세분화된 전문가(fine-grained experts)로 나누어 총 $N \times M$개의 전문가를 구성한다. 라우팅 레이어는 이 중 $k$개의 전문가를 선택한다.
- **World Expert**: 일반적인 지식을 학습하는 별도의 'World Expert'를 도입하여 모든 토큰이 이를 거치게 함으로써 기본 성능을 보장하고, 선택된 세분화 전문가들의 출력과 결합하여 최종 결과를 도출한다.

### 학습 절차

학습은 총 3단계로 진행된다.

1. **Multi-modal Pre-training**: 25억 개의 캡션 데이터와 5천만 개의 웹 데이터를 사용해 이미지-텍스트 정렬 및 기본 관계를 학습한다. 초반 25%는 Gated Cross-Attention만 학습하고, 이후 ViT의 후반부 파라미터를 해제하여 학습한다.
2. **Multi-task Continual Pre-training**: VQA, NLP, OCR, Detection 데이터(총 92M 샘플)를 사용하여 고수준의 시각 질문-답변 능력을 배양한다. 이때 이미지 해상도를 $224^2$에서 $448^2$로 높인다.
3. **Supervised Fine-tuning (SFT)**: 230만 개의 고품질 인스트럭션 데이터를 통해 지시 이행 능력을 활성화한다. 이 단계에서 Dense Baseline(EVLM-Chat)과 MoE 확장 모델(EVLM-MoE)이 생성된다.

## 📊 Results

### 벤치마크 성능

EVLM은 General VQA, Text-oriented VQA, General Multimodal 등 13개 벤치마크에서 평가되었다.

- **General VQA**: ScienceQA에서 EVLM-Chat(86.4%)과 EVLM-MoE(86.8%)는 가변 해상도를 사용하는 InfiMM-HD보다 우수한 성능을 보였다.
- **Text-oriented VQA**: AI2Diagram 데이터셋에서 EVLM-MoE(75.5%)와 EVLM-Chat(76.0%)가 높은 정확도를 기록하여, 복잡한 이미지 내 텍스트 세부 사항 이해 능력이 뛰어남을 입증하였다.
- **General Multimodal**: MME, MMB, MMB-CN, POPE 등에서 경쟁 모델 대비 우수한 성능을 보였으며, 특히 POPE에서 가장 좋은 성적을 거두어 환각(hallucination) 현상이 감소했음을 보여주었다.

### 이미지 및 비디오 캡셔닝

- **Image Dense Captioning**: GPT-4o와 Llama2-70B 등을 활용한 다단계 자동 캡션 파이프라인(생성 $\rightarrow$ 검증 $\rightarrow$ 재조합 $\rightarrow$ 스타일 수정)을 구축하여 학습시켰으며, 이를 통해 환각이 적고 매우 상세한 묘사가 가능한 결과물을 생성한다.
- **Video Captioning**: 비디오 이해를 위해 프레임 간 간섭을 방지하는 특수 어텐션 마스크를 설계하였다. GPT-4o를 심사위원으로 한 평가에서 Video-LLaVA 등 기존 모델보다 높은 정확도(7.22)와 상세도(5.73)를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

EVLM은 Cross-attention 메커니즘과 계층적 ViT 특징의 결합을 통해, 연산 효율성을 유지하면서도 고해상도 및 긴 시퀀스 데이터에 대한 인지 능력을 확보하였다. 특히 MoE 구조의 도입은 파라미터 수를 효율적으로 늘려 모델의 용량을 확장하는 데 기여하였다. 또한, 정교하게 설계된 데이터 정제 및 합성 파이프라인이 OCR 및 Dense Captioning 성능 향상에 결정적인 역할을 했음을 알 수 있다.

### 한계 및 논의사항

학습 과정 중 ImageNet-1K 기반의 세분화된 카테고리 분석 결과, 'Star(유명인)' 카테고리의 정확도가 상대적으로 낮게 나타났다. 이는 현재 사용된 사전 학습 데이터셋이 특정 세부 개념에 대해 충분한 정보를 제공하지 못하고 있음을 시사하며, 향후 데이터셋의 규모와 다양성을 더욱 확장해야 할 필요성을 보여준다.

## 📌 TL;DR

본 논문은 LLaVA 스타일의 MLLM이 가진 연산 오버헤드와 시각 인지 능력의 한계를 해결하기 위해, **Cross-attention**, **계층적 ViT 특징**, 그리고 **세분화된 MoE**를 결합한 **EVLM**을 제안한다. 제안된 모델은 효율적인 연산 구조를 통해 긴 시각적 시퀀스를 처리하면서도, 정교한 사전 학습과 파인튜닝을 통해 이미지/비디오 캡셔닝 및 VQA 작업에서 SOTA급 성능을 달성하였다. 이 연구는 효율적인 멀티모달 융합 구조가 향후 초고해상도 이미지 및 장편 비디오 이해 연구에 중요한 기반이 될 것임을 시사한다.
