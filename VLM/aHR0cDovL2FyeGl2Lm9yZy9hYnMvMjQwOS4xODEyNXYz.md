# LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities

Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, Xihui Liu (2024/2025)

## 🧩 Problem to Solve

최근 Large Multimodal Models (LMMs)는 2D 이미지 및 비디오 이해 능력에서 비약적인 발전을 이루었으나, 물리적 세계와 상호작용하는 데 필수적인 3D 공간 지능(3D spatial intelligence)은 여전히 부족한 상태이다. 기존의 3D LMM들은 주로 3D 포인트 클라우드(Point Clouds)를 입력으로 사용하는데, 이는 두 가지 결정적인 한계가 있다. 첫째, 2D 데이터에 비해 대규모 3D 비전-언어 데이터셋이 매우 부족하다. 둘째, 2D의 CLIP ViT와 같이 강력하고 일반화된 성능을 가진 사전 학습된 3D 인코더가 존재하지 않는다.

본 논문의 목표는 포인트 클라우드 대신 실제 로봇이나 에이전트가 주로 관찰하는 다시점 이미지(Multi-view images)를 기반으로 하며, 이미 검증된 2D LMM의 강력한 사전 지식(Priors)을 활용하여 2D 이해 능력을 유지하면서도 효율적으로 3D 공간 이해 능력을 부여하는 통합 프레임워크인 LLaVA-3D를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **"2D 패치에 3D 공간 정보를 직접 주입하여 3D 패치를 생성한다"**는 것이다. 구체적으로, 2D CLIP 패치 특징에 3D 위치 임베딩(3D position embeddings)을 더함으로써 2D LMM이 별도의 무거운 3D 인코더 없이도 3D 공간 맥락을 인식하게 만든다. 또한, 텍스트 기반의 좌표 출력 한계를 극복하기 위해 별도의 Grounding Decoder를 도입하여 정확한 3D Bounding Box를 직접 예측할 수 있게 설계하였다. 이를 통해 2D와 3D 능력을 동시에 갖춘 통합 아키텍처를 제안하며, 기존 3D LMM 대비 학습 수렴 속도를 3.5배 향상시켰다.

## 📎 Related Works

기존의 3D LMM 접근 방식은 크게 두 가지로 나뉜다. 하나는 LL3DA나 LEO와 같이 3D 포인트 클라우드 인코더를 직접 LLM에 연결하는 방식이며, 다른 하나는 3D-LLM이나 Scene-LLM처럼 다시점 이미지에서 2D 세그멘테이션 결과를 이용해 객체 중심의 3D 표현을 구축하는 방식이다. 그러나 포인트 클라우드 방식은 데이터 부족과 인코더 성능 문제에 직면해 있으며, 이미지 기반 방식은 세그멘테이션 과정이 복잡하고 계산 비용이 매우 높다는 한계가 있다.

LLaVA-3D는 이러한 한계를 극복하기 위해 ODIN의 아이디어를 차용하여, 2D 특징과 3D 위치 정보를 결합하는 방식을 사용한다. 이는 기존 3D LMM들이 의존하던 복잡한 오프라인 3D 객체 전처리를 생략하고, 2D LMM의 강력한 시각적-의미적 정렬(visual-semantic alignment) 능력을 그대로 활용한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

LLaVA-3D는 LLaVA-Video를 기본 모델로 하며, 다시점 이미지에서 추출된 2D 패치 특징에 3D 위치 정보를 결합하여 3D 패치를 생성하고, 이를 LLM에 입력하여 추론하는 구조이다. 시스템은 크게 **3D 패치 생성 $\rightarrow$ 3D 패치 풀링 $\rightarrow$ 3D 인지 인코딩 및 디코딩** 단계로 구성된다.

### 2. 3D Patch 생성

다시점 이미지 $\mathbf{X}$에서 CLIP 인코더를 통해 추출된 2D 패치 특징 $\mathbf{X}'_p$에 3D 위치 정보를 주입한다. 카메라의 내/외부 파라미터와 depth 맵을 이용해 각 패치의 3D 좌표 $\mathbf{P}$를 구하고, 이를 학습 가능한 2층 MLP 기반의 3D Position Encoding Layer를 통해 임베딩 $\mathbf{P}'$로 변환한다. 최종 3D 패치 $\mathbf{X}'_{3D}$는 다음과 같이 계산된다.

$$\mathbf{X}'_{3D} = \mathbf{X}'_p + \mathbf{P}'$$

### 3. 3D Patch Pooling

입력 이미지 수가 많아질 경우 LLM의 컨텍스트 길이 제한을 초과할 수 있으므로, 3D 공간 기반의 풀링 전략을 사용한다.

- **Voxelization Pooling**: 3D 공간을 격자(Voxel) 형태로 나누고, 동일 보셀 내의 패치들을 평균 풀링하여 토큰 수를 줄인다.
- **FPS Pooling**: Farthest Point Sampling을 통해 장면 전체를 대표하는 고정된 수의 토큰을 샘플링한다.

### 4. 3D-aware Position Encoding & Decoding

- **Encoding**: 3D 좌표가 포함된 지시문(예: "좌표 $(x, y, z)$에 있는 물체는 무엇인가?")을 처리하기 위해, 해당 좌표를 3D 위치 임베딩 층에 통과시켜 **3D Coordinate Token**으로 변환하여 LLM에 함께 입력한다.
- **Decoding (Grounding Decoder)**: LLM이 직접 텍스트로 좌표를 출력하는 것이 어렵다는 점을 해결하기 위해 설계되었다.
  - **Instance Queries**: 3D 패치에서 FPS로 샘플링된 쿼리들을 사용한다.
  - **Multi-scale 3D k-NN Attention**: 쿼리가 3D 패치의 가장 가까운 $k$개 이웃 특징만 참조하게 하여 효율성을 높이고, 다양한 스케일의 기하학적 정보를 캡처한다.
  - **Distance-Adaptive Self-Attention**: 쿼리 간의 유클리드 거리 $\mathbf{D}$를 기반으로 어텐션 바이어스를 추가하여 상대적 공간 관계를 모델링한다.
    $$\text{Attn}(\mathbf{Q}_i, \mathbf{K}_j, \mathbf{V}_j) = \text{Softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{C}} - \sigma \cdot \mathbf{D}\right)\mathbf{V}_j$$
  - **Box Head**: 최종 쿼리를 2층 MLP에 통과시켜 3D Bounding Box를 예측한다.

### 5. 학습 절차

학습은 두 단계로 진행된다.

- **Stage 1 (Multi-Task Instruction Tuning)**: 2D 비디오 데이터와 자체 구축한 **LLaVA-3D-Instruct-86K** (3D 데이터셋)를 사용하여 joint tuning을 수행한다. 이때 LLM, 위치 임베딩 층, Grounding Decoder를 동시에 학습시킨다.
- **Stage 2 (Decoder-only Fine-tuning)**: 나머지 모든 모듈을 동결(freeze)하고 Grounding Decoder와 위치 토큰만을 추가 학습시켜 3D 시각적 그라운딩(Visual Grounding) 성능을 극대화한다.

## 📊 Results

### 1. 실험 설정 및 데이터셋

- **데이터셋**: ScanQA, SQA3D, MMScan QA, OpenEQA (질의응답), Scan2Cap, MMScan Captioning (캡셔닝), ScanRefer, Multi3DRefer (그라운딩).
- **기준선**: 3D-LLM, LEO, Chat-Scene, GPT-4V, Gemini 등.

### 2. 주요 정량적 결과

- **3D 질의응답**: ScanQA, SQA3D에서 SOTA 성능을 달성하였으며, 특히 좌표 이해가 필수적인 MMScan QA에서 기존 모델(LL3DA, LEO) 대비 월등한 성능 향상을 보였다.
- **3D 캡셔닝**: Scan2Cap과 MMScan Captioning 모두에서 기존 SOTA를 크게 상회하였다. MMScan Captioning의 경우 색상(Color) 점수에서 49.5%, 디자인(Design) 점수에서 43.3% 향상되는 결과를 보였다.
- **3D 그라운딩**: 단일 단계(Single-stage) 방식임에도 불구하고 Multi3DRefer에서 49.8 Acc@0.25를 기록하며 SOTA를 달성하였다.
- **2D 성능 유지**: MVBench와 VideoMME 벤치마크에서 기반 모델인 LLaVA-Video와 거의 동일한 성능을 유지하여, 3D 능력을 추가하면서도 2D 능력이 훼손되지 않았음을 입증하였다.

### 3. 효율성 및 일반화

- **학습 속도**: LEO와 비교했을 때 동일 성능 도달까지의 학습 수렴 속도가 3.5배 더 빨랐다.
- **일반화**: LLaVA-1.5, InternVL2.5 등 다른 2D LMM을 기반으로 적용했을 때도 일관된 성능 향상이 관찰되었으며, 특히 비디오 LMM 기반일 때 성능이 가장 좋았다.

## 🧠 Insights & Discussion

본 논문은 3D 이해를 위해 반드시 거대한 3D 인코더나 포인트 클라우드가 필요한 것이 아니라, **잘 학습된 2D LMM의 표현력에 적절한 3D 위치 정보만 주입해도 충분하다**는 것을 보여주었다. 특히, 3D 패치 표현의 효과를 분석한 결과, 단순한 인식 작업(ScanQA)보다는 정밀한 위치 정보가 필요한 작업(MMScan QA, Scan2Cap)에서 성능 향상 폭이 훨씬 컸다. 이는 제안된 3D 패치가 실제로 모델에 공간 지능을 부여했음을 의미한다.

다만, 본 연구는 다시점 이미지로부터 depth 및 카메라 파라미터를 얻는 전처리가 필요하며, Grounding Decoder의 경우 2단계 학습을 거쳐야만 수렴한다는 점이 한계로 언급된다. 또한, 텍스트로 직접 좌표를 출력하는 것이 LLM에게 매우 어려운 과제임을 확인하였고, 이를 위해 외부 디코더를 도입한 것은 실용적인 해결책이지만 완전한 End-to-End 텍스트 생성 방식과는 거리가 있다.

## 📌 TL;DR

LLaVA-3D는 2D LMM의 강력한 성능을 유지하면서 3D 공간 지능을 부여하기 위해, 2D CLIP 패치에 3D 위치 임베딩을 더한 **'3D 패치'**와 정밀한 위치 예측을 위한 **'Grounding Decoder'**를 도입한 모델이다. 이 방식은 포인트 클라우드 기반 모델보다 학습 속도가 3.5배 빠르며, 다양한 3D 이해 및 그라운딩 벤치마크에서 SOTA를 달성하였다. 본 연구는 2D LMM을 3D 영역으로 확장하는 매우 효율적인 경로를 제시하였으며, 향후 로봇 조작(Manipulation) 및 내비게이션과 같은 물리적 상호작용 연구에 중요한 기초가 될 가능성이 높다.
