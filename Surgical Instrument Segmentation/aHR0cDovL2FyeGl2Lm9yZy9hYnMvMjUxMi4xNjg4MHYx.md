# Memory-Enhanced SAM3 for Occlusion-Robust Surgical Instrument Segmentation

Valay Bundele, Mehran Hosseinzadeh, Hendrik P. A. Lensch (2025)

## 🧩 Problem to Solve

본 논문은 내시경 수술 비디오에서 수술 도구 분할(Surgical Instrument Segmentation) 시 발생하는 여러 기술적 난제를 해결하고자 한다. 수술 영상은 잦은 가림(Occlusion), 급격한 움직임, 정반사 아티팩트(Specular artefacts), 그리고 도구가 화면 밖으로 나갔다가 다시 들어오는 장기적인 재진입(Long-term re-entry) 현상이 빈번하게 발생하여 객체의 일관된 추적과 ID 유지가 매우 어렵다.

최근 공개된 SAM3는 시공간 메모리(Spatio-temporal memory) 프레임워크를 통해 비디오 객체 분할에서 강력한 성능을 보이지만, 수술 장면에서는 다음과 같은 한계가 존재한다.

1. **무분별한 메모리 업데이트**: 저품질의 예측 결과(노이즈 섞인 마스크)가 메모리에 그대로 삽입되어 오류가 누적되는 현상이 발생한다.
2. **고정된 메모리 용량**: 고정된 시간적 위치 인코딩(Temporal positional encodings)으로 인해 메모리 용량이 제한되어, 긴 수술 과정 중 초기의 중요한 프레임 정보가 덮어씌워진다.
3. **가림 이후의 취약한 ID 복구**: 도구가 장시간 가려졌다가 다시 나타날 때, 뷰포인트 변화나 부분적 가시성으로 인해 잘못된 ID를 할당하는 Identity failure가 빈번하다.

따라서 본 연구의 목표는 추가 학습이 필요 없는(Training-free) 메모리 확장 및 개선 기법을 통해 가림에 강건하고 장기적인 추적이 가능한 ReMeDI-SAM3를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM3의 메모리 관리 체계를 정교화하여 오류 전파를 막고, 손실된 ID를 정확하게 복구하는 것이다. 주요 기여 사항은 다음과 같다.

1. **이중 메모리 설계(Dual-Memory Design)**: 고신뢰도 프레임만 저장하는 'Relevance-aware memory'와 가림 직전의 힌트를 저장하는 'Occlusion-aware memory'를 분리하여 메모리 오염을 방지하고 복구 능력을 높였다.
2. **특징 기반 재식별 모듈(Feature-based Re-identification Module)**: 다중 스케일 외관 특징(Multi-scale appearance features)과 시간적 투표(Temporal voting)를 결합하여 가림 해제 후 도구의 ID를 명시적으로 검증하고 교정한다.
3. **메모리 용량 확장 전략(Memory Capacity Expansion)**: 시간적 위치 인코딩의 구간별 보간(Piecewise interpolation)을 통해 모델의 재학습 없이도 더 긴 시간적 컨텍스트를 유지할 수 있도록 메모리 크기를 확장했다.

## 📎 Related Works

**SAM 기반 시각적 추적**
SAM은 제로샷 이미지 분할을 가능케 했으며, SAM2는 메모리 기반 패러다임을 통해 비디오로 확장되었다. 이후 DAM4SAM, SAMURAI, HiM2SAM 등이 메모리 신뢰성 향상 및 모션 가이딩을 통해 추적 성능을 개선했다. 최신 모델인 SAM3는 오픈 보캐블러리 검출, 분할, 추적을 통합하였으나, 여전히 장기 가림 상황에서의 ID 유지 및 재식별 문제에 취약하다.

**수술 도구 분할**
ISINet, TAFPNet 등 전문 모델들이 제안되었으며, 최근에는 SAM 및 SAM2를 수술 도메인에 맞게 미세 조정(Fine-tuning)하거나 LoRA 어댑터를 사용하는 연구(SurgicalSAM, SP-SAM 등)가 진행되었다. 또한 SurgSAM2, MA-SAM2, SAMed-2 등은 메모리 기반 비디오 분할을 통해 장기 추적을 시도했다.

**차별점**
기존 연구들이 주로 도메인 적응(Domain adaptation)이나 단순한 메모리 프루닝에 집중한 반면, ReMeDI-SAM3는 가림 이후의 **명시적인 ID 복구(Re-identification)**와 **확장 가능한 메모리 용량**이라는 두 가지 핵심 문제에 집중한 최초의 SAM 기반 확장 모델이라는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 메모리 구조

ReMeDI-SAM3는 SAM3의 메모리 뱅크를 두 개의 상보적인 구성 요소로 분할하여 관리한다. 전체 메모리 크기 $M$을 $M/2$씩 나누어 다음과 같이 운용한다.

- **Relevance-Aware Memory ($U_{rel}$)**: 안정적인 추적을 위해 신뢰도가 높은 프레임만 저장한다. 신뢰도 점수 $r_t$를 다음과 같이 정의한다.
  $$r_t = s_t \cdot c_t$$
  여기서 $s_t$는 objectness score, $c_t$는 quality score이다. $r_t \ge \tau_{rel}$ 조건을 만족하는 최근 $M/2$개의 프레임만 저장하여 메모리 오염을 방지한다.
- **Occlusion-Aware Memory ($U_{occ}$)**: 도구가 다시 나타났을 때(Disocclusion) ID 복구를 돕기 위한 전용 메모리이다. 모든 과거 프레임을 저장하는 'Unconditional Buffer'에서 $r_t \ge \tau_{occ}$ (단, $\tau_{occ} < \tau_{rel}$)를 만족하는 최근 $M/2$개의 프레임을 선택하여 채운다. 가림 직전의 낮은 신뢰도 프레임이라도 중요한 외관 힌트를 포함할 수 있음을 이용한 설계이다.

### 2. 메모리 용량 확장 (Memory Capacity Expansion)

SAM3는 기본적으로 7개의 고정된 시간적 위치 인코딩 $\{p_0, \dots, p_6\}$을 사용한다. 본 논문은 이를 확장하기 위해 **구간별 보간(Piecewise interpolation)** 방식을 제안한다.

- 경계 부분인 $\tilde{p}_0 = p_0$와 $\tilde{p}_{M-1} = p_6$는 고정하여 시간적 우선순위(Temporal priors)를 유지한다.
- 내부 구간 $(p_1, \dots, p_5)$에 대해서만 선형 보간을 수행하여 $M-2$개의 새로운 인코딩을 생성한다.
  $$\tilde{p}_k = (1 - \alpha_k)p_{\lfloor u_k \rfloor} + \alpha_k p_{\lceil u_k \rceil}, \quad \alpha_k = u_k - \lfloor u_k \rfloor$$
  이 방식은 재학습 없이도 더 많은 프레임을 인덱싱할 수 있게 하여 장기적인 컨텍스트 유지를 가능케 한다.

### 3. 특징 기반 재식별 (Feature-based Re-ID)

가림 해제 후 ID 드리프트를 방지하기 위해 외관 기술자(Appearance descriptors)를 이용한 검증 단계를 거친다.

- **Reference Feature Bank ($B_i$)**: 각 도구 클래스 $i$에 대해 신뢰도가 높은 프레임에서 추출한 다중 스케일 특징 $\left\{f_{i,t,l}\right\}$을 저장한다. 특징은 백본 피처 맵 $F_{t,l}$을 예측된 마스크 $M_{i,t}$ 영역 내에서 평균 풀링하여 계산한다.
  $$f_{i,t,l} = \frac{1}{|M_{i,t}|} \sum_{x \in M_{i,t}} F_{t,l}(x)$$
- **Temporal Voting**: 단일 프레임의 불안정성을 줄이기 위해 $K$개의 연속된 프레임 윈도우에서 재식별을 수행한다.
  - **Self-similarity ($s_{self}$)**: 복구된 객체와 해당 클래스 $B_i$ 간의 코사인 유사도 평균.
  - **Cross-instrument similarity ($s_{other}$)**: 다른 클래스 $B_j$와의 최대 유사도 평균.
- **ID 결정 규칙**: 다음 조건을 만족하면 ID를 수용한다.
  $$s_{self} - s_{other} \ge \delta_{sim} \quad \text{and} \quad \text{IoU} \le \delta_{iou}$$
  만약 다른 클래스 $j$에 대해 $s_{other} - s_{self} \ge \delta_{-sim}$가 성립하면 ID를 $j$로 재할당하며, 둘 다 아니면 오탐지로 간주하여 폐기한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis17 (8개 시퀀스), EndoVis18 (검증 세트 4개 비디오).
- **지표**: Challenge IoU, IoU, mean class IoU (mcIoU).
- **설정**: Zero-shot 설정으로 평가하였으며, RTX 4090 GPU 1대를 사용하였다.

### 정량적 결과

ReMeDI-SAM3는 Vanilla SAM3 대비 괄목할 만한 성능 향상을 보였다.

- **EndoVis17**: Challenge IoU 약 7.2% 상승, mcIoU 약 6.9% 상승.
- **EndoVis18**: IoU 약 6% 상승, **mcIoU 약 16% 상승**.
특히 mcIoU의 큰 폭의 상승은 Re-ID 모듈이 도구가 없는 상황에서 발생하는 가짜 양성(False Positive) 예측을 효과적으로 억제했음을 보여준다. 이는 기존의 학습 기반(Training-based) 접근 방식보다도 우수한 성능이다.

### 정성적 결과 및 분석

- **ID 유지 능력**: SAM3는 가림 이후 다른 도구가 나타나면 기존 ID를 잘못 부여하는 Identity drift 현상이 심했으나, ReMeDI-SAM3는 가림 해제 후 정확한 ID를 복구하거나 불필요한 예측을 억제하는 모습을 보였다.
- **Ablation Study**:
  - Relevance-aware memory(RM) 도입 시 IoU 3.5% 상승.
  - Memory Expansion(ME) 추가 시 1.0% 추가 상승.
  - Re-ID 모듈 추가 시 1.2% 추가 상승.
  - Occlusion-aware memory(OM) 적용 시 최종적으로 1.5% 추가 상승하여 총 7.2%의 IoU 이득을 얻었다.
- **보간 방식 비교**: 단순 균등 보간(Uniform interpolation)보다 제안된 구간별 보간(Piecewise interpolation)이 더 높은 성능을 보였는데, 이는 학습된 경계 시간 우선순위를 보존했기 때문이다.

## 🧠 Insights & Discussion

**강점**
본 논문은 SAM3라는 강력한 파운데이션 모델을 기반으로, 도메인 특화 학습 없이(Training-free) 메모리 구조와 재식별 로직만으로 수술 영상의 난제인 가림 문제를 해결했다는 점이 매우 인상적이다. 특히 단순한 성능 수치 향상을 넘어, mcIoU 분석을 통해 ID 보존 능력을 논리적으로 입증하였다.

**한계 및 비판적 해석**
논문에서 명시했듯이, ReMeDI-SAM3는 오탐지를 줄이기 위해 매우 보수적인 업데이트 전략을 취한다. 이로 인해 도구가 재진입했을 때 즉각적으로 반응하지 못하고, 시각적 증거가 충분히 쌓일 때까지 **재검출이 지연(Delayed re-detection)**되는 트레이드-오프가 존재한다. 이는 실시간성이 극도로 중요한 수술 보조 시스템에서는 잠재적인 단점이 될 수 있다.

또한, 제안된 $\tau_{rel}, \tau_{occ}, \delta_{sim}$ 등의 하이퍼파라미터가 실험적으로 설정되었는데, 다양한 수술 환경(다른 조명, 다른 도구 세트)에서도 이 값들이 일반화될 수 있을지에 대한 분석이 추가로 필요해 보인다.

## 📌 TL;DR

본 연구는 SAM3의 메모리 관리 방식을 개선하여 수술 도구 분할의 가림 문제를 해결한 **ReMeDI-SAM3**를 제안한다. 신뢰도 기반의 이중 메모리 구조, 구간별 보간을 통한 메모리 확장, 그리고 다중 스케일 특징 기반의 Re-ID 모듈을 통해 재학습 없이도 강력한 ID 유지 능력을 확보했다. 결과적으로 EndoVis17/18 데이터셋에서 기존 SAM3 및 학습 기반 모델들을 뛰어넘는 성능을 보였으며, 이는 향후 수술 로봇의 실시간 추적 및 워크플로우 분석 연구에 중요한 기초가 될 것으로 기대된다.
