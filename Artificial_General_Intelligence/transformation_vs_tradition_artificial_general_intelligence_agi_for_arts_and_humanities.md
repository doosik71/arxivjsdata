# Transformation vs Tradition: Artificial General Intelligence (AGI) for Arts and Humanities

Zhengliang Liu, Yiwei Li, Qian Cao, Junwen Chen, Tianze Yang, Zihao Wu, John Gibbs, Khaled Rasheed, Ninghao Liu, Gengchen Mai, and Tianming Liu (2023)

## 🧩 Problem to Solve

본 논문은 인공 일반 지능(Artificial General Intelligence, AGI)의 급격한 발전이 전통적으로 인간의 고유 영역으로 간주되었던 예술 및 인문학 분야에 미치는 영향과 그로 인해 발생하는 문제들을 다룬다.

전통적으로 예술과 인문학은 인간의 주관성, 창의성, 그리고 깊은 정서적 경험을 바탕으로 한다. 그러나 최근 Large Language Models(LLMs)와 창의적 이미지 생성 시스템의 등장으로 기계의 계산 능력과 인간의 창의성 사이의 경계가 모호해지고 있다. 이에 따라 AGI가 이러한 문화적으로 중요한 도메인에 배포될 때 발생할 수 있는 책임감 있는 활용 문제, 특히 사실성(Factuality), 독성(Toxicity), 편향성(Bias), 그리고 공공 안전에 관한 비판적 분석이 필요하게 되었다. 본 논문의 목표는 텍스트, 그래픽, 오디오, 비디오 등 다양한 모달리티에서 AGI의 응용 사례를 종합적으로 분석하고, 잠재적 위험을 완화하기 위한 전략을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AGI가 예술 및 인문학에 적용되는 현재의 기술적 지형을 체계적으로 조사하고, 기술적 가능성과 윤리적 책임 사이의 균형점을 모색했다는 점에 있다.

중심적인 아이디어는 AGI를 인간의 창의성을 대체하는 도구가 아닌, 인간의 표현력을 확장하고 증강하는 파트너로 정의하는 것이다. 이를 위해 텍스트, 그래픽, 비디오/오디오의 세 가지 주요 축으로 나누어 최신 기술적 진보를 분석하였으며, AGI 시스템의 신뢰성을 높이기 위한 사실성 평가, 독성 제거(Detoxification), 편향성 탐지 등의 구체적인 완화 전략을 제안하였다.

## 📎 Related Works

논문은 생성형 AI의 발전 과정을 다음과 같은 흐름으로 설명하며 기존 접근 방식을 검토한다.

1. **초기 모델**: Hidden Markov Models 및 Gaussian Mixture Models와 같은 통계적 모델에서 시작하여 GANs, VAEs와 같은 딥러닝 기반 생성 모델로 진화하였다.
2. **Transformer의 등장**: 2017년 Transformer 아키텍처의 등장은 대규모 사전 학습 모델(GPT-3, GPT-4 등)의 가능성을 열었으며, 이는 텍스트 생성의 패러다임을 바꾸었다.
3. **이미지 생성의 진화**: Deep Dream(2015)이나 Neural Style Transfer와 같은 초기 시도는 단순한 스타일 변환에 그쳤으나, 이후 GANs를 거쳐 최근의 Diffusion Models로 이어지며 완전히 새로운 고품질 콘텐츠 생성이 가능해졌다.
4. **차별점**: 기존의 AIGC(AI-Generated Content) 관련 연구들이 주로 특정 모델의 성능 향상이나 단일 모달리티의 생성 능력에 집중했다면, 본 논문은 이를 '예술 및 인문학'이라는 거시적인 학문적/문화적 맥락에서 통합적으로 분석하고 책임감 있는 배포(Responsible Deployment)라는 윤리적 관점을 강력하게 결합했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 논문이 아니라, AGI의 기술적 구성 요소와 응용 분야를 분석하는 서베이(Survey) 성격의 논문이다. 각 모달리티별 핵심 방법론은 다음과 같다.

### 1. Text Analysis and Generation

- **Transformer Architecture**: Recurrent layers 대신 Self-Attention 메커니즘을 사용하여 시퀀스 내 단어 간의 관계를 병렬적으로 캡처한다.
- **BERT (Bidirectional Encoder Representations from Transformers)**: 양방향 문맥을 고려하는 Masked Language Model 목적 함수를 통해 텍스트의 의미를 깊게 이해한다.
- **Autoregressive Models**: 이전 토큰들을 기반으로 다음 토큰을 예측하는 방식으로, GPT 시리즈가 대표적이다. 텍스트 생성 시 Beam search, Greedy decoding, Probabilistic sampling 등의 디코딩 전략을 사용한다.

### 2. Graphics Analysis and Generation

- **GANs (Generative Adversarial Networks)**: Generator와 Discriminator가 서로 경쟁하는 Minimax Game 구조를 통해 실제와 구별 불가능한 이미지를 생성한다.
- **Diffusion Models**: 가우시안 노이즈를 점진적으로 추가하는 forward process와 이를 다시 복원하는 reverse process를 학습한다.
  - **DDPM (Denoising Diffusion Probabilistic Models)**: 마르코프 체인을 통해 노이즈를 제거하지만 계산 비용이 매우 높다.
  - **DDIM (Denoising Diffusion Implicit Models)**: Non-Markovian 과정을 도입하여 샘플링 단계(Step)를 획기적으로 줄여 생성 속도를 높였다.
  - **Latent Diffusion Models (LDMs)**: 픽셀 공간이 아닌 압축된 Latent Space에서 확산 과정을 수행함으로써 계산 효율성을 극대화하고 고해상도 이미지를 생성한다. Stable Diffusion이 이 방식의 대표 사례이다.

### 3. Video and Audio Generation

- **Video Diffusion Models (VDM)**: 이미지 확산 모델을 비디오 도메인으로 확장하여 시간적 일관성(Temporal Consistency)을 유지하며 영상을 생성한다.
- **Zero-shot Text-to-Video**: 추가 학습 없이 사전 학습된 텍스트-이미지 모델의 Latent Space 내에서 모션 다이내믹스를 수정하여 비디오를 생성하는 방식(예: Text2Video-Zero)이 논의된다.

## 📊 Results

본 논문은 정량적인 실험 수치보다는 AGI가 실제 예술/인문학 분야에서 어떻게 적용되고 있는지에 대한 정성적 결과와 사례 분석을 제시한다.

- **인문학적 적용**: GPT-4V를 사용하여 고지도(Historical Map)에서 지명과 좌표를 추출하는 작업(Mapkurator system)에서 유의미한 성능을 보였다.
- **디자인 및 예술**: Adobe Firefly를 이용한 폰트 디자인, LLaVA를 통한 건축 디자인 분석, Midjourney와 DALL-E 3를 이용한 초현실적 사진 생성 등의 사례가 제시되었다.
- **인기 추세**: Google Trends 및 Subreddit 데이터를 통해 Midjourney, Stable Diffusion, DALL-E의 대중적 관심도가 급증하고 있음을 확인하였다.
- **한계점**: AI 생성 이미지에서 인간의 손가락 개수가 틀리게 생성되는 문제나, 원격 탐사 이미지에서 지리적 레이아웃이 부정확하게 나타나는 등의 Factuality 문제가 여전히 존재함을 지적하였다.

## 🧠 Insights & Discussion

### 1. AI 창의성의 본질에 대한 논의

논문은 AI의 창의성이 알고리즘과 데이터의 패턴 인식에 기반한 것임을 강조한다. 인간의 창의성은 개인적 경험, 감정, 상상력에서 비롯되지만, AI는 기존 데이터의 조합에 의존하므로 완전히 새로운(Novel) 아이디어를 창조하는 데 한계가 있으며, 특히 예술의 핵심인 '정서적 깊이'와 '공감'을 재현할 수 없다는 비판적 시각을 유지한다.

### 2. 책임감 있는 AGI (Responsible AGI)를 위한 제언

AGI의 확산에 따른 위험을 완화하기 위해 다음과 같은 전략적 방향을 제시한다.

- **사실성(Factuality)**: ROUGE, BLEU와 같은 지표나 모델 기반 메트릭을 통해 생성된 텍스트의 사실성을 평가하고, "Truthful AI"와 같은 엄격한 표준을 설정해야 한다.
- **공공 안전(Public Safety)**: 딥페이크를 통한 허위 정보 확산, AI 기반 스피어 피싱(Spear Phishing) 등의 위협에 대응하기 위해 AI 기반 분류기(Classifier)를 통한 탐지 기술을 개발해야 한다.
- **독성 및 편향성(Toxicity & Bias)**: RLHF(Reinforcement Learning from Human Feedback)를 통해 모델을 인간의 가치관에 정렬(Alignment)시키고, 추론 시점(Inference-time)에서 프롬프트 학습이나 디코딩 제어(Decoding-time steering)를 통해 유해 콘텐츠 생성을 억제해야 한다.

## 📌 TL;DR

본 논문은 AGI가 예술 및 인문학 분야에 가져온 혁신적인 변화와 그에 따른 윤리적/기술적 위험을 분석한 포괄적인 보고서이다. 텍스트의 Transformer, 이미지의 Diffusion 모델 등 최신 AGI 기술이 어떻게 창작 과정을 보조하고 있는지 설명하는 동시에, AI가 가진 사실성 결여, 편향성, 독성 문제를 지적한다. 결론적으로 AGI를 인간의 대체재가 아닌 창의성을 증강하는 도구로 활용하기 위해서는 기술적 보완(RLHF, 탐지 모델 등)과 더불어 인간 중심의 가치를 우선시하는 다각적인 협력이 필수적임을 강조한다.
