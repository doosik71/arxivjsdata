# VLM-Vac: Enhancing Smart Vacuums through VLM Knowledge Distillation and Language-Guided Experience Replay

Reihaneh Mirjalili, Michael Krawez, Florian Walter and Wolfram Burgard (2024)

## 🧩 Problem to Solve

본 연구는 스마트 로봇 청소기가 실제 가정 환경과 같이 복잡하고 다양한 환경에서 자율적으로 판단하고 행동하는 능력을 향상시키는 것을 목표로 한다. 기존의 로봇 청소기는 단순히 바닥 전체를 덮는 방식으로 작동하지만, 이는 액체나 끈적이는 물질을 퍼뜨리거나 가치 있는 물건을 흡입하는 위험이 있다.

이를 해결하기 위해 딥러닝 기반의 오염물 감지 방법들이 제안되었으나, 이러한 방법들은 수동으로 주석을 단 데이터셋(manually annotated datasets)에 크게 의존하며, 이는 비용이 많이 들고 실제 환경의 다양성을 모두 반영하기 어렵다는 한계가 있다. 최근 Vision-Language Model (VLM)이 뛰어난 제로샷(zero-shot) 객체 인식 능력을 보여주었으나, VLM을 실시간으로 쿼리하는 것은 계산 비용이 매우 높고 에너지 소모가 커서 로봇과 같은 엣지 디바이스(edge device)에 직접 배포하는 것은 불가능에 가깝다. 따라서 본 논문은 VLM의 강력한 지식을 유지하면서도 계산 효율성을 확보하여, 로봇이 실시간으로 물체를 '흡입(suck)'할지 '회피(avoid)'할지를 결정할 수 있는 시스템을 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고비용의 VLM 지식을 효율적인 소형 모델로 전이하는 Knowledge Distillation (KD)과, 동적인 환경에서 과거의 지식을 잊지 않도록 돕는 Language-Guided Experience Replay를 결합한 VLM-Vac 프레임워크를 제안하는 것이다.

주요 기여 사항은 다음과 같다.

1. VLM의 제로샷 인식 능력을 활용하여 스마트 청소기의 자율성을 높이는 VLM-Vac 프레임워크를 제안한다.
2. 계산 비용이 큰 VLM에서 소형 모델인 YOLOv8n으로 지식을 전이하는 KD 프로세스를 구현하여 실시간 추론 효율성을 높였다.
3. 동적 환경에서의 Continual Learning (CL) 문제를 해결하기 위해, 텍스트 임베딩 기반의 새로운 experience replay 방법을 제안하여 catastrophic forgetting(치명적 망각)을 방지했다.
4. TurtleBot 4를 이용하여 실제 가정 환경을 모사한 데이터셋을 구축하고 이를 통해 제안 방법론을 검증하였다.

## 📎 Related Works

기존의 청소 로봇 연구들은 saliency detection, spectral analysis 또는 YOLO 계열의 데이터 기반 방법론을 사용해왔다. 그러나 이러한 모델들은 학습 데이터셋의 크기가 현대의 VLM에 비해 매우 작아 open-world 시나리오에서의 인식 능력이 부족하다. 또한, 일부 연구에서 LLM을 활용해 청소 추천을 제공하는 시도가 있었으나, 모델 크기 감소나 환경 변화에 따른 적응(adaptation) 문제는 다루지 않았다.

Continual Learning 분야에서는 EWC(Elastic Weight Consolidation)와 같은 정규화 방법이나 experience replay 방식이 연구되었다. 특히 replay 기반 방식은 도메인 변화(domain-incremental learning)가 빈번한 환경에서 더 효과적이라는 점이 알려져 있다. 본 연구는 이러한 배경을 바탕으로, 단순한 시각적 특징이 아닌 언어적 기술자(language-based descriptors)를 활용한 샘플링이 더 강건한 이미지 표현이 될 수 있다는 점에 착안하여 이를 experience replay에 접목하였다.

## 🛠️ Methodology

VLM-Vac의 전체 구조는 VLM으로부터의 지식 증류(Knowledge Distillation)와 언어 기반의 경험 재생(Experience Replay)이라는 두 가지 핵심 구성 요소로 이루어져 있다.

### 1. Action-based Object Classification을 위한 Knowledge Distillation

실시간 추론을 위해 소형 모델인 YOLOv8n을 Student 모델로 사용하고, GPT-4o를 Teacher 모델(VLM)로 사용한다.

- **작동 흐름**: YOLOv8n이 이미지 $I$를 인식했을 때, 설정된 신뢰도 임계값(Confidence Threshold)보다 낮은 확신을 가지면 VLM에 쿼리를 보낸다.
- **VLM 쿼리**: VLM은 프롬프트 $p$와 이미지 $I$를 받아 다음과 같은 텍스트 설명 $t(c, q, a, f)$를 출력한다.
  $$t(c, q, a, f) = \text{VLM}(p, I)$$
  여기서 $c$는 물체 카테고리, $q$는 수량, $a$는 행동 클래스(suck/avoid), $f$는 바닥 유형(floor type)을 의미한다.
- **Grounding**: VLM이 텍스트로 설명한 물체의 정확한 위치를 파악하기 위해 open-vocabulary object detection 모델인 OWL-ViT를 사용하여 bounding box $B$를 생성한다.
- **Experience Pool**: 생성된 이미지 $I_{new}$, 텍스트 설명 $t_{new}$, bounding box $B_{new}$는 experience pool $E$에 저장되며, 이후 YOLOv8n을 미세 조정(fine-tuning)하는 데 사용된다.

### 2. Language-Based Experience Replay를 통한 Continual Learning

로봇이 새로운 환경(예: 다른 방의 바닥 패턴)을 접할 때 발생하는 catastrophic forgetting을 방지하기 위해, 단순한 전체 학습(cumulative training) 대신 언어 기반의 샘플링 전략을 사용한다.

- **언어 임베딩 추출**: VLM이 제공한 텍스트 정보($c, q, a, f$)를 바탕으로 언어 임베딩 $e_{new}$를 생성한다.
  $$e_{new} = \text{Embedding}(c_{new}, q_{new}, a_{new}, f_{new})$$
- **K-means 클러스터링**: 생성된 임베딩들을 k-means 알고리즘을 통해 그룹화한다. 목적 함수는 다음과 같다.
  $$\min_{\mu} \sum_{k=1}^{K} \sum_{e \in C_k} \|e - \mu_k\|^2$$
  여기서 $C_k$는 $k$번째 클러스터, $\mu_k$는 해당 클러스터의 중심점이다.
- **버퍼 구성**: 각 클러스터에서 무작위로 하위 집합을 선택하여 경험 재생 버퍼 $B$를 구성한다. 이렇게 하면 특정 클래스나 배경에 치우치지 않은 균형 잡힌 데이터 믹스를 구성할 수 있어, 새로운 데이터를 학습하면서도 과거의 지식을 유지할 수 있다.

## 📊 Results

### 실험 설정

- **하드웨어**: TurtleBot 4 Pro, OAK-D-PRO RGB-D 카메라, NVIDIA RTX 6000 GPU 워크스테이션.
- **데이터셋**: 3가지 바닥 패턴과 12가지 물체 카테고리를 포함한 2,500장의 이미지.
- **비교 대상**: Cumulative Learning (전체 데이터 학습), Naive Fine-tuning (최근 데이터만 학습).

### 주요 결과

1. **클러스터링 성능**: 언어 기반 클러스터링의 평균 클래스 순도(mean class purity)는 $93.11\%$로, 시각 기반 클러스터링의 $74.12\%$보다 훨씬 높았다. 특히 시각 기반 방식은 작은 물체(반지, 부스러기 등)를 배경 특징에 따라 잘못 그룹화하는 경향이 있었으나, 언어 기반 방식은 배경이 달라도 동일 물체를 정확히 그룹화했다.
2. **학습 성능 (F1 Score)**: Naive Fine-tuning은 환경이 바뀔 때마다 성능이 급격히 떨어지는 catastrophic forgetting 현상을 보였다. 반면, 제안된 Language-based ER은 Cumulative Learning과 유사한 높은 $F1$ 스코어를 유지하였다.
3. **에너지 효율성**: Cumulative Learning은 데이터셋이 커짐에 따라 GPU 에너지 소모가 기하급수적으로 증가했다. 반면 Language-based ER은 Cumulative Learning 대비 에너지 소모를 $53\%$ 줄이면서도 유사한 성능(Mean $F1$: $0.913$ vs $0.930$)을 달성했다.
4. **VLM 쿼리 감소**: 시간이 지남에 따라 VLM에 요청하는 쿼리의 비율이 지속적으로 감소하는 추세를 보였다. 이는 VLM의 지식이 YOLOv8n으로 성공적으로 증류(distillation)되었음을 입증한다.

## 🧠 Insights & Discussion

본 연구는 VLM의 강력한 상식 추론 능력을 엣지 디바이스에 적합한 소형 모델로 효율적으로 전이시킬 수 있음을 보여주었다. 특히, 이미지의 시각적 특징만으로 샘플링하는 대신 VLM이 생성한 텍스트 설명을 임베딩하여 클러스터링하는 방식이 소형 객체 인식 및 도메인 적응에 훨씬 유리하다는 점을 밝혀낸 것이 큰 성과이다.

**한계 및 논의 사항:**

- **장기적 가소성(Plasticity) 문제**: 논문에서는 9일간의 실험을 진행했으나, 딥러닝 모델이 장기간 학습할 때 발생하는 가소성 저하 문제에 대해서는 언급만 하였을 뿐 직접적으로 해결하지 않았다.
- **OOD 탐지 부재**: 모델이 확신을 가지고 잘못된 예측을 내리는 경우를 방지하기 위한 Out-of-Distribution (OOD) 탐지 알고리즘의 도입이 필요하다.
- **데이터 다양성**: 실험에 사용된 물체와 배경의 종류가 제한적이므로, 더 방대한 실제 환경에서의 검증이 필요하다.

## 📌 TL;DR

VLM-Vac은 고비용의 VLM(GPT-4o) 지식을 소형 모델(YOLOv8n)로 증류하여 스마트 청소기의 실시간 객체 인식 성능을 높인 프레임워크이다. 특히 텍스트 임베딩 기반의 경험 재생(Experience Replay) 기법을 통해 에너지 소모를 53% 절감하면서도, 새로운 환경에 적응할 때 발생하는 치명적 망각 문제를 효과적으로 해결했다. 이 연구는 VLM을 활용한 로봇의 온디바이스 학습(on-device learning) 및 효율적인 지식 전이 전략 수립에 중요한 기여를 한다.
