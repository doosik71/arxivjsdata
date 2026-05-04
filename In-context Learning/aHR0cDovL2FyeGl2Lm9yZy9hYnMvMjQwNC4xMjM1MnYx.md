# Point-In-Context: Understanding Point Cloud via In-Context Learning

Mengyuan Liu, Zhongbin Fang, Xia Li, Joachim M. Buhmann, Xiangtai Li, Chen Change Loy (2024)

## 🧩 Problem to Solve

본 논문은 자연어 처리(NLP)와 2D 이미지 처리 분야에서 성공적으로 도입된 In-Context Learning(ICL) 패러다임을 3D Point Cloud 분석 영역으로 확장하고자 한다. 기존의 대규모 모델들은 다중 작업(multitasking) 수행 능력이 뛰어나지만, 새로운 작업에 적응시키기 위해 전체 파라미터를 미세 조정(full fine-tuning)하거나 효율적인 튜닝 기법을 사용하는 것은 여전히 막대한 계산 자원을 요구한다.

반면 ICL은 모델의 파라미터 업데이트 없이 입력 단계에서 도메인 특정 입력-출력 쌍(prompts)을 제공함으로써 모델이 새로운 작업에 적응하게 하는 효율적인 방법이다. 그러나 3D Point Cloud 데이터는 2D 이미지와 달리 구조화되지 않은(unordered) 특성을 가지며, 기존의 Masked Point Modeling(MPM) 프레임워크를 ICL에 그대로 적용할 경우 위치 임베딩(position embedding)으로 인한 정보 누설(information leakage) 문제가 발생할 수 있다. 따라서 본 연구의 목표는 이러한 기술적 난제를 해결하여 3D Point Cloud 이해를 위한 범용적인 ICL 프레임워크인 Point-In-Context(PIC)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 3D Point Cloud 분석을 위한 최초의 ICL 프레임워크인 PIC를 설계한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **Joint Sampling (JS) 모듈**: 입력과 타겟 포인트 클라우드 간의 인덱스 일관성을 유지함으로써 정보 누설을 방지하고, 비정형 데이터인 포인트 클라우드를 토큰 시퀀스로 변환할 때 발생할 수 있는 정렬 문제를 해결하였다.
2.  **PIC-Generalist (PIC-G)**: 재구성(reconstruction), 노이즈 제거(denoising), 정합(registration), 파트 분할(part segmentation) 등 다양한 작업을 하나의 통일된 입력-출력 공간(XYZ 좌표)에서 처리하는 범용 모델을 제안하였다.
3.  **PIC-Segmenter (PIC-S)**: 고정된 레이블 좌표 할당의 한계를 극복하기 위해 **In-Context Labeling**과 **In-Context Enhancing** 전략을 도입하여, 학습하지 않은 새로운 데이터셋에 대해서도 높은 일반화 성능을 보이는 분할 특화 모델을 제안하였다.

## 📎 Related Works

### 3D Point Cloud Analysis
PointNet, DGCNN과 같은 포인트 기반/그래프 기반 방법론부터 Point Transformer와 같은 트랜스포머 기반 방법론까지 다양한 연구가 진행되었다. 최근에는 State Space Models(SSM)를 활용한 PCM 등의 연구도 등장하였다. 그러나 이러한 기존 방법론들은 대부분 단일 작업(single task)을 위해 설계되었으며, ICL을 통해 파라미터 업데이트 없이 작업을 전환하는 능력은 탐구되지 않았다.

### Masked X Modeling
BERT의 MLM이나 MAE의 MIM처럼, 3D 영역에서도 Point-BERT나 PointMAE와 같은 Masked Point Modeling(MPM)이 제안되었다. 본 논문은 MPM의 파이프라인을 따르지만, 이를 단순히 사전 학습 도구로 쓰는 것이 아니라 ICL의 메커니즘으로 활용하여 태스크 프롬프트를 통해 출력을 제어한다는 점에서 차별점을 가진다.

### Multi-Task Learning (MTL)
기존의 3D MTL 연구들은 주로 공유 백본(shared backbone) 위에 작업별 헤드(task-specific heads)를 여러 개 두는 방식을 사용하였다. 하지만 PIC는 모든 작업을 통일된 좌표 회귀 문제로 정의하여 별도의 헤드 설계 없이 프롬프트만으로 작업을 수행하는 방식을 취한다.

## 🛠️ Methodology

### 1. Point-In-Context-Generalist (PIC-G)

PIC-G는 MPM 프레임워크를 기반으로 하며, 입력과 타겟 쌍을 통해 모델이 마스킹된 영역을 복원하도록 학습한다.

#### Joint Sampling (JS) 모듈
기존 MPM은 마스킹된 패치의 중심점 좌표를 위치 임베딩으로 사용하는데, 이는 추론 시 타겟 좌표를 미리 아는 꼴이 되어 정보 누설을 야기한다. 이를 해결하기 위해 JS 모듈은 다음과 같이 동작한다.
- 입력 포인트 클라우드에서 $N_C$개의 중심점을 샘플링하고 그 인덱스를 기록한다.
- 동일한 인덱스를 사용하여 입력과 타겟 포인트 클라우드 모두에서 패치를 동시에 샘플링한다.
- 이를 통해 입력 토큰 시퀀스와 타겟 토큰 시퀀스의 순서가 완벽히 정렬되어, 타겟의 위치 임베딩 없이도 학습이 가능해진다.

#### 모델 구조 및 학습
백본으로 Encoder-Decoder 구조의 Transformer를 사용하며, 두 가지 베이스라인을 탐구한다.
- **PIC-G-Sep**: 입력과 마스킹된 타겟을 병렬로 입력하고 나중에 특징을 병합한다.
- **PIC-G-Cat**: 입력과 타겟을 하나로 연결(concatenate)하여 전체적으로 마스킹 후 입력한다.

손실 함수로는 예측된 패치 $P$와 정답 패치 $G$ 사이의 $\ell_2$ Chamfer Distance (CD)를 사용한다.
$$L_{CD}(P, G) = \frac{1}{|P|} \sum_{p \in P} \min_{g \in G} \|p-g\|_2^2 + \frac{1}{|G|} \sum_{g \in G} \min_{p \in P} \|p-g\|_2^2$$

### 2. Point-In-Context-Segmenter (PIC-S)

분할 작업에서 PIC-G는 고정된 레이블 좌표 맵을 사용하므로 일반화 능력이 떨어진다. 이를 해결하기 위해 PIC-S는 다음 전략을 사용한다.

#### In-Context Labeling
고정된 좌표 대신 **동적 컨텍스트 레이블**을 사용한다.
1.  3D 공간을 균등하게 분할하여 레이블 포인트 뱅크 $B$를 생성한다.
2.  각 샘플을 학습할 때마다 뱅크 $B$에서 무작위로 레이블 포인트 $B_i$를 선택하여 파트 카테고리에 할당한다.
3.  프롬프트와 쿼리가 동일한 파트 시맨틱을 공유하도록 구성하여, 모델이 좌표 자체가 아닌 프롬프트 내의 매핑 관계를 학습하게 한다.

#### In-Context Enhancing
모델의 강건성을 높이기 위해, 깨끗한 포인트 클라우드 $P_j$에 무작위 손상 함수 $r(\cdot)$를 적용한 쌍 $Q_j^r = \{r(P_j), P_j\}$을 추가하여 학습시킨다. 이는 모델이 공간적 컨텍스트만을 이용해 복원하는 능력을 키우게 한다.

#### 손실 함수 및 추론
PIC-S는 CD 손실에 Smooth-$\ell_1$ 손실을 추가하여 정밀한 회귀를 유도한다.
$$L_{Seg}(P, G) = L_{CD}(P, G) + L_{Smooth-\ell_1}(P, G)$$
추론 시에는 프롬프트에서 사용된 레이블 포인트 뱅크 $B_j$를 기준으로 가장 가까운 포인트의 클래스를 할당하는 방식으로 분할 결과를 도출한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ShapeNet, ShapeNetPart, Human3D, BEHAVE, AKB-48.
- **지표**: 재구성, 노이즈 제거, 정합 작업은 Chamfer Distance (CD)를, 파트 분할은 mIoU를 사용한다.
- **비교 대상**: 작업별 특화 모델(Task-Specific), 공유 백본 기반 다중 작업 모델(Multitask), PointMAE 등.

### 주요 결과
1.  **범용 성능**: PIC-G와 PIC-S는 단 한 번의 학습으로 4가지 작업 모두에서 다중 작업 모델들을 압도하며 SOTA 성능을 달성하였다. 특히 파트 분할에서 PIC-S는 개별 특화 모델보다 더 높은 성능을 보였다 (Table 1).
2.  **분할 특화 성능**: Human & Object Segmentation 벤치마크에서 PIC-S는 모든 모델 중 가장 높은 mIoU를 기록하였다 (Table 2).
3.  **일반화 능력**:
    - **Out-of-Distribution**: 학습에 사용되지 않은 ModelNet40 데이터셋의 정합 작업에서 지도 학습 모델보다 뛰어난 성능을 보였다.
    - **One-Shot Generalization**: 학습 데이터에 포함되지 않은 AKB-48 데이터셋에 대해 단 하나의 프롬프트만 제공했을 때, PIC-S는 타 모델 대비 월등한 일반화 성능을 보였다.
4.  **Ablation Study**:
    - JS 모듈이 없을 경우 성능이 급격히 하락하여, 입력-타겟 간 시퀀스 정렬이 필수적임이 증명되었다.
    - 마스크 비율이 70%일 때 가장 좋은 성능을 보였으며, 이는 고도의 희소성(sparsity)이 숨겨진 특징 학습에 중요함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 3D 포인트 클라우드 분야에 ICL이라는 새로운 패러다임을 성공적으로 도입하였다. 특히 작업별 헤드 없이 좌표 회귀만으로 다중 작업을 수행한다는 점과, 동적 레이블링을 통해 미학습 데이터셋에 대한 제로샷/원샷 일반화 가능성을 열었다는 점이 매우 높게 평가된다.

### 한계 및 논의사항
- **프롬프트 의존성**: 실험 결과 프롬프트의 품질(예: CD-aware selection)에 따라 성능 편차가 크게 나타났다. 이는 실제 적용 시 최적의 프롬프트를 어떻게 선택하거나 생성할 것인가라는 새로운 과제를 던진다.
- **계산 복잡도**: 트랜스포머 기반의 구조와 다수의 포인트 패치를 처리해야 하므로, 포인트 수가 급격히 증가할 경우 메모리 및 계산 비용 문제가 발생할 수 있다.
- **레이블 뱅크의 설계**: 레이블 포인트 뱅크 $B$를 균등 분할로 생성하였으나, 데이터의 분포에 따라 더 효율적인 뱅크 구성 방법이 존재할 가능성이 있다.

## 📌 TL;DR

본 논문은 3D 포인트 클라우드 분석을 위한 최초의 In-Context Learning 프레임워크인 **Point-In-Context (PIC)**를 제안한다. 정보 누설을 방지하는 **Joint Sampling** 모듈과 일반화 성능을 극대화하는 **동적 레이블링(In-Context Labeling)** 전략을 통해, 파라미터 업데이트 없이 프롬프트만으로 재구성, 노이즈 제거, 정합, 분할 등 다양한 작업을 수행할 수 있음을 보였다. 특히 PIC-S 모델은 학습하지 않은 데이터셋에 대해서도 뛰어난 원샷 일반화 능력을 보여, 향후 3D 비전 모델의 유연한 확장성과 범용성 연구에 중요한 기여를 할 것으로 기대된다.