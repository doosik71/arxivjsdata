# VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE
Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh & Kai-Wei Chang

## 🧩 Problem to Solve
시각과 언어를 결합한 다양한 태스크(예: 이미지 캡셔닝, 시각 질의응답, 시각 추론)는 시각 정보 시스템의 추론 능력을 평가하는 중요한 척도입니다. 이러한 태스크들은 시스템이 객체, 속성, 관계, 행동 등 이미지의 복잡한 세부 의미를 이해하고, 이를 자연어와 연결하여 grounding하는 것을 요구합니다. 기존 모델들은 종종 특정 태스크에 맞춰 설계되어 일반화가 어렵거나, 시각 및 언어 정보를 효과적으로 융합하지 못하는 한계가 있었습니다. 본 연구는 다양한 시각-언어 태스크에 적용 가능하며, 이미지와 텍스트 간의 풍부한 의미론적 상호작용을 포착하는 간단하고 유연한 통합 모델을 제안합니다.

## ✨ Key Contributions
*   **VisualBERT 프레임워크 제안**: BERT 아키텍처를 기반으로 텍스트와 이미지 영역을 동시에 처리하는 간단하고 유연한 통합 모델 VisualBERT를 제안합니다.
*   **시각적으로 Grounded된 사전 학습 목표**: 이미지 캡션 데이터(COCO)를 활용하여 시각 정보를 기반으로 하는 두 가지 언어 모델링 목표(Masked Language Modeling with Image, Sentence-Image Prediction)를 제안하여 모델이 이미지와 텍스트 간의 연관성을 학습하도록 합니다.
*   **최첨단 성능 달성**: VQA, VCR, NLVR$_{2}$, Flickr30K 등 네 가지 주요 시각-언어 태스크에서 기존 SOTA 모델과 필적하거나 능가하는 성능을 달성하며, 기존 모델보다 훨씬 간단한 구조를 가집니다.
*   **암묵적인 시각적 Grounding 능력 입증**: 명시적인 supervision 없이도 VisualBERT가 언어 요소를 이미지 영역에 grounding할 수 있음을 실험적으로 보여줍니다. 특히, 동사와 그 인자에 해당하는 이미지 영역 간의 구문론적 관계까지 추적하는 능력을 입증합니다.
*   **심층적인 Attention 분석**: VisualBERT의 어텐션 메커니즘이 단어와 이미지 영역 간의 정렬을 어떻게 학습하고 개선하는지 정량적 및 정성적으로 분석합니다.

## 📎 Related Works
*   **시각-언어 태스크 모델**: VQA (Antol et al., 2015; Goyal et al., 2017), Textual Grounding (Kazemzadeh et al., 2014; Plummer et al., 2015), Visual Reasoning (Suhr et al., 2019; Zellers et al., 2019) 등 다양한 태스크와 이를 해결하기 위한 모델들(Yang et al., 2016; Anderson et al., 2018; Jiang et al., 2018)이 언급됩니다. 기존 모델들은 대부분 특정 태스크에 맞춰 설계되었습니다.
*   **시각적 세부 의미 이해**: 이미지 내 객체 관계 모델링의 중요성 (Johnson et al., 2015)과 Visual Genome (Krishna et al., 2017)의 속성 주석 활용 (Anderson et al., 2018), 주의 메커니즘을 통한 객체 관계 모델링 (Santoro et al., 2017; Norcliffe-Brown et al., 2018; Cadene et al., 2019), 명시적 그래프 구축을 통한 관계 인코딩 (Li et al., 2019) 등의 연구들이 참조됩니다.
*   **Transformer 기반 언어 모델**: 자연어 처리 분야의 범용 언어 인코더 학습 트렌드를 이끄는 BERT (Devlin et al., 2019), GPT (Radford et al., 2018; 2019), ELMo (Peters et al., 2018) 등이 영감을 주었습니다.
*   **동시 연구**: 유사한 아이디어를 탐구하는 두 가지 동시 연구가 언급됩니다.
    *   **VideoBERT (Sun et al., 2019)**: 비디오와 언어의 결합 표현 학습에 Transformer를 적용.
    *   **ViLBERT (Jiasen et al., 2019)**: BERT와 유사한 아키텍처를 사용하여 이미지와 텍스트의 결합 표현을 학습하지만, 시각 및 언어에 별도의 Transformer를 사용하고 서로에게만 어텐션하여 두 배 많은 파라미터를 가집니다.

## 🛠️ Methodology
VisualBERT는 BERT의 Transformer 아키텍처를 기반으로 시각 정보를 통합합니다.
1.  **입력 임베딩 구성**:
    *   **텍스트 임베딩($E$)**: BERT와 동일하게 토큰 임베딩($e_t$), 세그먼트 임베딩($e_s$), 위치 임베딩($e_p$)의 합으로 구성됩니다.
    *   **시각 임베딩($F$)**: 사전 학습된 객체 감지기(예: Faster R-CNN)로부터 추출된 바운딩 박스 영역별 특징을 사용합니다. 각 시각 임베딩($f \in F$)은 다음 세 가지 합으로 구성됩니다.
        *   $f_o$: 바운딩 영역의 시각적 특징 표현 (CNN으로 계산).
        *   $f_s$: 해당 임베딩이 이미지 임베딩임을 나타내는 세그먼트 임베딩.
        *   $f_p$: 위치 임베딩. (단어와 바운딩 영역 간의 정렬 정보가 입력으로 제공될 경우, 정렬된 단어들의 위치 임베딩 합을 사용하며, 그렇지 않으면 기본적으로 0).
2.  **Transformer 통한 공동 처리**: 텍스트 임베딩($E$)과 시각 임베딩($F$)이 함께 BERT의 멀티레이어 Transformer 스택에 입력됩니다. Self-attention 메커니즘을 통해 모델은 텍스트와 이미지 요소 간의 암묵적인 정렬을 발견하고 새로운 공동 표현을 구축합니다. (만약 텍스트와 시각 임베딩의 차원이 다르면, 시각 임베딩을 텍스트 임베딩과 동일한 차원으로 투영합니다.)
3.  **VisualBERT 학습 절차**: BERT와 유사하게 세 단계로 구성됩니다.
    *   **Task-Agnostic 사전 학습**: COCO 이미지 캡션 데이터셋에서 두 가지 시각적으로 grounded된 언어 모델링 목표로 학습합니다.
        *   **Masked Language Modeling with Image**: 텍스트 입력의 일부 단어를 마스킹하고, 남은 텍스트와 시각적 맥락을 기반으로 마스킹된 단어를 예측합니다. 이미지 영역에 해당하는 벡터는 마스킹하지 않습니다.
        *   **Sentence-Image Prediction**: 주어진 텍스트 세그먼트(두 개의 캡션으로 구성)가 이미지와 일치하는지 여부를 예측합니다. 50% 확률로 무작위 캡션을 사용하여 불일치 상황을 만듭니다.
    *   **Task-Specific 사전 학습**: 다운스트림 태스크에 파인튜닝하기 전에, 해당 태스크의 데이터로 "Masked Language Modeling with Image" 목표를 사용하여 모델을 추가 학습합니다. 이는 새로운 도메인에 모델을 적응시키는 데 도움이 됩니다.
    *   **파인튜닝**: 특정 태스크에 맞는 입력, 출력 레이어 및 목표 함수를 추가하여 사전 학습된 파라미터로 모델을 파인튜닝하여 태스크 성능을 최대화합니다.

## 📊 Results
VisualBERT는 네 가지 시각-언어 태스크에서 강력한 성능을 보여주며, 기존 SOTA 모델과 비교하여 더 간단한 구조로도 우수하거나 경쟁력 있는 결과를 달성했습니다.
*   **VQA 2.0 (Visual Question Answering)**: 비교 가능한 설정에서 VisualBERT가 Pythia v0.1 및 v0.3을 크게 능가하며 70.80% (Test-Dev) 및 71.00% (Test-Std)의 정확도를 기록했습니다. COCO 사전 학습이 없는 VisualBERT는 70.18%로 성능이 저하되어 사전 학습의 중요성을 보여줍니다.
*   **VCR (Visual Commonsense Reasoning)**: R2C 모델(동일 자원 사용)보다 훨씬 간단함에도 불구하고 VisualBERT w/o COCO Pre-training이 R2C를 큰 폭으로 능가했습니다. 전체 VisualBERT 모델은 Q→AR(질문, 답변, 추론 모두 맞춤)에서 Test 52.4%로 추가적인 성능 향상을 보였습니다. COCO와 VCR의 도메인 차이에도 불구하고 COCO 사전 학습이 유의미한 도움을 주었습니다.
*   **NLVR$_{2}$ (Natural Language for Visual Reasoning)**: VisualBERT와 그 변형 모델(VisualBERT w/o Early Fusion, VisualBERT w/o COCO Pre-training) 모두 이전 SOTA 모델인 MaxEnt를 큰 폭으로 뛰어넘었습니다. VisualBERT는 Test-P 67.0%, Test-U 67.3%로 가장 좋은 성능을 달성했습니다.
*   **Flickr30K Entities (Region-to-Phrase Grounding)**: VisualBERT는 SOTA 모델인 BAN을 능가하며, R@1 Test에서 71.33%, R@5 Test에서 84.98%, R@10 Test에서 86.51%의 정확도를 기록했습니다. 이 태스크에서는 early fusion이 없는 모델과 전체 모델 간의 성능 차이가 크지 않아, 얕은 아키텍처로도 충분할 수 있음을 시사했습니다.

## 🧠 Insights & Discussion
*   **사전 학습의 중요성**: Task-agnostic 사전 학습(특히 COCO 데이터와 이미지 및 캡션 동시 사용)과 초기 시각-언어 융합(early fusion)이 VisualBERT의 강력한 성능에 가장 중요한 설계 요소임을 ablation study를 통해 확인했습니다. BERT 초기화도 중요하지만, COCO 사전 학습을 통해 많은 유용한 grounded language 특징을 학습할 수 있음을 보여주었습니다.
*   **암묵적 Grounding 능력**: VisualBERT는 명시적인 grounding supervision 없이도 어텐션 헤드를 통해 개체 grounding(단어가 해당하는 바운딩 영역에 주의를 기울이는 것)을 매우 정확하게 수행할 수 있습니다. 특히, 어텐션 정확도는 상위 레이어에서 증가하며, 모델이 하위 레이어에서 두 입력을 통합하고 상위 레이어에서 정렬을 인식하는 과정을 보여줍니다.
*   **구문론적 Grounding**: VisualBERT의 어텐션 헤드는 구문론적 관계(예: "pobj", "nsub", "dobj" 의존 관계)를 통해 grounding 정보를 전달할 수 있습니다. 이는 모델이 암묵적으로, 그리고 supervision 없이 동사와 그 인자를 시각적 요소에 연결하는 능력을 가짐을 시사합니다.
*   **점진적인 정렬 개선**: 정성적 분석(어텐션 가중치 시각화)을 통해 VisualBERT가 Transformer 레이어를 거치면서 이미지와 텍스트 간의 모호한 정렬을 점진적으로 개선하고 해소하는 과정을 관찰할 수 있었습니다.
*   **한계 및 향후 연구**: VisualBERT는 단순하지만 강력한 성능을 보입니다. 향후 이미지 전용 태스크(예: 장면 그래프 파싱)로의 확장 가능성, Visual Genome이나 Conceptual Caption과 같은 더 큰 캡션 데이터셋으로의 사전 학습, 그리고 다양한 도메인에 대한 일반화 능력을 탐구하는 것이 유효한 방향입니다.

## 📌 TL;DR
VisualBERT는 BERT와 Faster R-CNN을 통합하여 텍스트와 이미지 영역을 동시에 처리하는 간단한 Transformer 기반 모델입니다. COCO 이미지 캡션 데이터에서 마스킹된 언어 모델링 및 문장-이미지 예측 목표로 사전 학습되어, 명시적인 grounding 없이도 언어 요소를 이미지 영역에 암묵적으로 정렬하는 능력을 학습합니다. VQA, VCR, NLVR$_{2}$, Flickr30K 등 다양한 시각-언어 태스크에서 SOTA 또는 경쟁력 있는 성능을 달성하며, 사전 학습과 초기 시각-언어 융합이 핵심적인 역할을 함을 입증했습니다.