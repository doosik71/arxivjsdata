# VLAS: VISION-LANGUAGE-ACTION MODEL WITH SPEECH INSTRUCTIONS FOR CUSTOMIZED ROBOT MANIPULATION

Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang

---

## 🧩 Problem to Solve

기존 Vision-Language-Action (VLA) 모델들은 주로 텍스트 기반 명령에 의존하여 로봇 조작을 수행하며, 인간-로봇 상호작용에 더 자연스러운 음성 모달리티를 간과하고 있습니다. 전통적인 음성 통합 방식은 별도의 자동 음성 인식(ASR) 시스템을 사용하는데, 이는 모델을 복잡하게 만들고 오류 전파를 야기하며, 화자 식별과 같은 비의미론적 정보를 손실시켜 맞춤형 작업 수행에 중요한 정보를 누락시킵니다. 따라서 개인의 신체 능력이나 주관적인 선호도에 따라 로봇이 더 접근 가능하고 맞춤화될 수 있도록, 음성 모달리티를 VLA 모델에 직접 통합하여 자연스러운 상호작용과 맞춤형 작업을 지원하는 방법이 필요합니다.

## ✨ Key Contributions

- **VLAS 제안:** 외부 음성 인식 시스템 없이 로봇 조작을 위해 음성 명령을 직접 통합하는 최초의 종단 간(end-to-end) VLA 모델을 제안하여 로봇과의 보다 자연스러운 의사소통을 가능하게 합니다.
- **Voice RAG 패러다임 설계:** 개인별 특정 지식이 필요한 맞춤형 작업을 VLAS가 효과적으로 처리할 수 있도록 Voice Retrieval-Augmented Generation (RAG) 패러다임을 설계했습니다.
- **VLAS-Base 도입:** 널리 사용되는 VLM인 LLaVA를 음성 명령을 수용하도록 확장한 VLAS-Base를 소개했습니다. 이 모델은 로봇 정책 모델 외에도 음성 입력을 사용하는 다른 다운스트림 작업에도 활용될 수 있습니다.
- **새로운 데이터셋 구축:** 음성 명령 튜닝을 지원하기 위해 SQA (Speech Question Answering) 및 CSI (CALVIN with Speech Instructions) 두 가지 새로운 데이터셋을 구축하여 공개했습니다.

## 📎 Related Works

- **Vision-Language Models (VLMs):** FLAN-PaLM, InstructGPT, LLaMA, Mamba와 같은 대규모 언어 모델(LLM)과 OpenFlamingo, BLIP-2, LLaMA-Adapter, LLaVA, Cobra와 같은 시각-언어 모델들이 존재합니다. 특히 LLaVA는 시각 명령 튜닝을 통해 뛰어난 성능을 보였으나, 음성 명령 지원이 부족했습니다. GPT-4o, Gemini, VITA와 같은 최근 모델들은 오디오 정보의 직접적인 통합을 탐구하고 있습니다.
- **Vision-Language-Action Models (VLAs):** 로봇 공학 분야에서는 VLM을 활용하여 고수준 작업 계획(예: PaLM-E, SayCan) 또는 로봇 동작 직접 생성(예: RT-2, Roboflamingo, OpenVLA)에 초점을 맞추고 있습니다. 현재 대부분의 VLA는 텍스트 명령과 시각적 관찰에 중점을 두며, 촉각이나 깊이 정보와 같은 추가 모달리티를 통합하는 연구도 있지만, 음성 모달리티 통합 연구는 부족하며, 대부분 외부 ASR 도구를 사용합니다. 본 연구는 음성 통합을 종단 간 방식으로 발전시킵니다.

## 🛠️ Methodology

VLAS는 인간의 음성 명령($s$)과 시각적 관찰($O$)을 입력으로 받아 로봇 동작($a$)을 직접 생성하는 종단 간 VLA 모델입니다.

- **VLAS 아키텍처:**
  - **전체 프레임워크:**
    - 입력 이미지($O$)와 음성 명령($s$)은 각각 Vision Encoder ($\text{Emb}_v$)와 Speech Encoder ($\text{Emb}_s$)를 통해 임베딩 토큰 시퀀스로 변환됩니다.
    - 추론 시, Voice RAG 모듈의 출력($\text{RAG}(s)$)은 Text Tokenizer ($\text{Tok}_l$)를 통해 임베딩 토큰으로 변환됩니다.
    - 시각 및 음성 토큰은 각각 Multi-Layer Perceptrons ($\text{MLP}_v$, $\text{MLP}_s$)를 통해 LLaVA의 통합 언어 공간으로 투영됩니다.
    - 모든 임베딩 토큰은 연결되어 LLM 백본의 입력으로 사용됩니다:
      $$ \text{Emb}(s,O) = \text{concat}(\text{MLP}\_s(\text{Emb}\_s(s)), \text{Tok}\_l(\text{RAG}(s)), \text{MLP}\_v(\text{Emb}\_v(O))) $$
    - LLM 백본은 이 임베딩을 받아 자기회귀(autoregressive) 방식으로 예측된 동작($p(a|\text{Emb}(s,O))$)을 생성하며, 이는 detokenizer를 통해 연속적인 값으로 변환됩니다.
  - **네트워크 백본:** LLaVA (Vision Transformer로 CLIP, LLM 백본으로 Vicuna)를 기반으로 합니다.
  - **음성 인코더:** Whisper 인코더를 사용하여 음성 명령을 80-bin 멜-스펙트로그램으로 변환한 후 은닉 상태 시퀀스를 생성합니다. 계산 부담을 줄이기 위해 시간 차원에서 5의 감소 계수로 리셰이프(reshape)하고, MLP를 통해 텍스트 및 시각 토큰과 공유되는 의미 공간으로 투영합니다.
  - **Voice RAG (Retrieval-Augmented Generation):**
    - 음성 명령의 음성 지문(voiceprint)을 화자 식별 모듈로 추출합니다.
    - 이 음성 지문은 외부 데이터베이스에서 사용자 지정 지식(개인 선호도, 소유물 등)을 검색하는 키로 사용됩니다.
    - 검색된 정보는 배경 지식으로 통합되어 시각 및 음성 토큰과 함께 LLM에 전달되어 개인화된 작업 수행을 돕습니다.
  - **액션 토큰화:** 로봇의 연속적인 7차원 동작 값($[x, y, z, \phi, \theta, \psi, g]$)을 256개의 이산화된 bin으로 나누고, LLM 어휘에서 사용 빈도가 낮은 256개 토큰을 동작 토큰으로 재활용하여 학습 레이블로 사용합니다.
- **데이터 수집:**
  - **SQA (Speech Question Answering) 데이터셋:** LLaVA의 시각 명령 튜닝 데이터셋에서 이미지-텍스트 질문-답변 쌍을 추출하여, 텍스트 질문을 ESPnet (VITS TTS 모델)를 통해 1,152개 이상의 다양한 음성으로 185K 오디오 질문으로 변환했습니다.
  - **CSI (CALVIN with Speech Instructions) 데이터셋:** CALVIN 로봇 조작 데이터셋의 389개 텍스트 명령을 동일한 TTS 모델을 사용하여 500개의 다른 음성으로 약 194K 오디오 명령으로 변환했습니다. 훈련 시 기존 텍스트 명령 샘플의 절반을 합성된 음성 명령으로 무작위 교체하여 사용했습니다.
- **VLAS 훈련 패러다임 (3단계):**
  - **단계 I: 음성 정렬 (Speech Alignment):** LibriSpeech-360 데이터셋으로 음성 인코더와 LLM 백본 사이의 MLP 레이어만 미세 조정하여 음성과 텍스트 간의 양상 정렬을 수행합니다.
  - **단계 II: 음성 질문 답변 미세 조정 (Speech Question Answering Fine-tuning):** SQA, LLaVA의 VQA, LibriSpeech-100 데이터셋으로 미세 조정합니다. 이미지 및 음성 인코더를 제외한 모든 네트워크 구성 요소를 업데이트하며, 이 단계 후 VLAS-Base 모델을 얻습니다.
  - **단계 III: 로봇 조작 미세 조정 (Robot Manipulation Fine-tuning):** CSI 로봇 조작 데이터셋으로 행동 복제(behavior cloning)를 통해 VLAS-Base를 미세 조정합니다.

## 📊 Results

- **CALVIN 벤치마크 (로봇 조작):**
  - VLAS는 텍스트 또는 음성 명령 모두에서 MCIL, HULC, RT-1과 같은 기존 모델보다 훨씬 우수한 성능을 달성했습니다.
  - 텍스트 명령 사용 시, VLAS($\text{LH-5}=56.6\%$)는 베이스라인 VLA($\text{LH-5}=58.2\%$)와 비슷한 성능을 보였습니다.
  - 음성 명령 사용 시, VLAS($\text{LH-5}=54.6\%$)는 외부 ASR(Whisper large-v2)을 사용하는 VLA+ASR($\text{LH-5}=40.2\%$) 및 Roboflamingo+ASR($\text{LH-5}=48.3\%$)보다 월등히 뛰어난 성능을 보였습니다.
  - 실제 음성 명령 사용 시에도 VLAS($\text{LH-5}=51.3\%$)는 강한 성능을 유지하며, VLA 베이스라인 대비 0.19% 포인트 차이로 약간 뒤처지는 수준이었습니다.
- **맞춤형 작업 벤치마크 (Voice RAG 효과):**
  - 객체 소유권, 사용자 선호도, 복합 작업 등 개인화된 작업에서 VLA 베이스라인은 평균 성공률 20% 미만으로 저조했습니다.
  - VLAS (음성 입력 + Voice RAG)는 평균 86.5% 이상의 성공률을 달성하여 압도적인 성능을 보였습니다 (예: 소유권 작업 94.7%, 선호도 작업 84.6%).
  - RAG 모듈 제거 시 VLAS의 성능은 크게 저하되었고(평균 16.0%), VLA에 RAG 모듈을 통합했을 때 성능이 크게 향상되어(평균 82.0%) Voice RAG의 효과를 입증했습니다.
- **실제 UR5 로봇 팔 실험:** VLAS 모델을 실제 UR5 로봇 팔에 배포하여 화자 ID 기반의 컵 소유권 식별과 같은 맞춤형 조작 작업을 성공적으로 수행했습니다.
- **VLAS-Base 파운데이션 모델 분석:**
  - **일반 멀티모달 벤치마크:** VLAS-Base는 LLaVA와 거의 동일한 성능을 달성하며, 음성 모달리티 도입이 시각-언어 이해 성능을 저하시키지 않음을 보여주었습니다.
  - **음성 이해 벤치마크 (LibriSpeech ASR, SGQA Q&A):** LibriSpeech ASR에서는 Whisper large-v2와 유사한 성능(WER 2.79%)을, SGQA (이미지-음성 Q&A)에서는 LLaVA(정답 텍스트)에는 미치지 못하지만 BLIP-2보다 우수한 성능(50.8%)을 보여 음성 명령 처리 능력이 뛰어남을 확인했습니다.

## 🧠 Insights & Discussion

VLAS는 외부 ASR 시스템 없이 음성 명령을 종단 간으로 이해함으로써 로봇 제어 파이프라인을 단순화하고 성능을 향상시키는 중요한 진전을 이루었습니다. 특히, 원시 음성에서 화자 지문과 같은 비의미론적 정보를 직접 활용하고 Voice RAG 패러다임을 통해 개인 맞춤형 지식을 통합함으로써, "내 컵을 집어줘"와 같이 모호하고 개인화된 명령을 로봇이 성공적으로 수행할 수 있게 되었습니다. 이는 인간-로봇 상호작용의 자연스러움과 맞춤화 가능성을 크게 높이는 결과입니다.

VLAS-Base는 음성 기능을 통합하면서도 기존 LLaVA의 강력한 시각-언어 이해 능력을 유지하여, 다른 멀티모달 LLM 연구에 중요한 기반 모델로 활용될 잠재력을 가집니다. 맞춤형 벤치마크에서의 압도적인 성능은 Voice RAG가 복잡한 음성 명령을 이해하고 실행하는 데 있어 핵심적인 역할을 한다는 것을 명확히 보여줍니다.

향후 연구에서는 VLAS의 실패 사례(주로 선호도 및 복합 작업의 2단계)에 대한 정책 모델 아키텍처 및 훈련 프로세스 개선이 필요합니다. 또한 인간 음성이나 환경 소리의 다른 보조 정보를 탐색하여 로봇이 더욱 복잡한 작업을 이해하고 완료할 수 있도록 하는 방향으로 확장될 수 있습니다. 추론 효율성을 위해 도입된 음성 스펙트로그램 다운샘플링과 다단계 예측 전략은 성능과 속도 사이의 최적 균형을 제공함을 확인했습니다.

## 📌 TL;DR

VLAS는 외부 음성 인식 시스템 없이 음성 명령을 직접 이해하여 로봇을 조작하는 최초의 종단 간 Vision-Language-Action (VLA) 모델입니다. 이 모델은 음성 지문과 같은 비의미론적 정보를 활용하고, Voice Retrieval-Augmented Generation (RAG) 패러다임을 통해 개인 맞춤형 지식을 통합하여 "나의 컵 들어줘"와 같은 사용자 맞춤형 작업을 탁월하게 수행합니다. CALVIN 벤치마크에서 기존 VLA 및 ASR 기반 모델을 능가하는 성능을 보였으며, 실제 로봇 환경에서도 성공적으로 시연되어 자연스럽고 개인화된 로봇 상호작용의 가능성을 제시합니다.
