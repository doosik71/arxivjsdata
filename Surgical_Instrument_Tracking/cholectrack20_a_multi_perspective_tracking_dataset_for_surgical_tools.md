# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

Chinedu Innocent Nwoye et al. (2025)

## 🧩 Problem to Solve

수술 비디오에서의 도구 추적(Tool Tracking)은 기술 평가, 안전 구역 추정 및 인간-기계 협업과 같은 컴퓨터 보조 중재(computer-assisted interventions)를 발전시키는 데 필수적이다. 그러나 현재 이 분야의 AI 적용을 제한하는 핵심 문제는 맥락 정보가 풍부한 데이터셋의 부족이다.

기존의 데이터셋들은 지나치게 일반적인 추적 정식화(tracking formalizations)에 의존하고 있어, 도구가 카메라 시야 밖으로 벗어나거나(Out-of-Camera View, OOCV) 신체 밖으로 나가는(Out-of-Body, OOB) 것과 같은 수술 특유의 동적인 상황을 제대로 포착하지 못한다. 이로 인해 임상적으로 유의미한 궤적(trajectory)을 얻기 어렵고 실제 수술 환경에 적용하는 데 유연성이 떨어진다. 또한, 기존 방법론들은 연기, 반사, 출혈과 같은 수술실의 시각적 챌린지(visual challenges) 상황에서 성능이 급격히 저하되는 한계를 보인다. 본 논문의 목표는 이러한 한계를 극복하기 위해 다중 클래스 및 다중 도구 추적을 위한 전문 데이터셋인 **CholecTrack20**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 도구 추적을 위한 새로운 관점의 정식화를 도입하고, 이를 기반으로 한 대규모 데이터셋을 구축한 것이다.

1. **Multi-Perspective Tracking의 정의**: 도구의 궤적을 단순히 시각적 가시성으로만 정의하지 않고, (1) Intraoperative(수술 내), (2) Intracorporeal(체내), (3) Visibility(가시성)라는 세 가지 관점에서 재정의하여 임상적으로 더 의미 있는 궤적 추적을 가능하게 하였다.
2. **CholecTrack20 데이터셋 구축**: 복강경 담낭 절제술 수술 비디오 20개를 대상으로, 1 fps 간격으로 어노테이션을 수행하여 35,000개 이상의 프레임과 65,000개의 도구 인스턴스를 포함하는 고정밀 데이터셋을 제공한다.
3. **포괄적인 어노테이션**: 단순한 위치 정보뿐만 아니라 도구 범주, ID, 조작자(Operator), 수술 단계(Phase), 그리고 장면의 시각적 챌린지까지 포함하여 데이터의 풍부함을 더했다.
4. **SOTA 벤치마크 분석**: 최신 객체 검출 및 추적 알고리즘을 통해 데이터셋의 유효성을 검증하고, 현재 기술 수준이 임상 적용을 위해서는 여전히 부족함을 정량적으로 제시하였다.

## 📎 Related Works

기존의 객체 검출 및 추적 연구는 COCO, KITTI, MOTChallenge와 같은 일반 데이터셋을 중심으로 발전해 왔으며, 이를 수술 도구 추적에 적용하려는 시도가 있었다. 하지만 수술 환경은 출혈, 연기, 급격한 움직임, 가변적인 조명 등으로 인해 일반적인 환경보다 훨씬 복잡하다.

기존의 수술 도구 추적 접근 방식은 크게 세 가지로 나뉜다:

- **Single Object Tracking (SOT)**: 단일 객체에 집중한다.
- **Multi-Class Tracking (MCT)**: 클래스당 하나의 도구만 추적한다.
- **Multi-Object Tracking (MOT)**: 모든 도구를 하나의 클래스로 취급한다.

그러나 실제 수술에서는 여러 클래스의 여러 도구가 역동적으로 상호작용하는 **Multi-Class Multi-Object Tracking (MC-MOT)** 상황이 발생하며, 특히 도구가 시야에서 사라졌다가 다시 나타나거나 도구가 교체되는 상황을 처리하는 능력이 부족하다. 기존 데이터셋들은 이러한 세밀한 궤적 관점(granularity)을 제공하지 못하며, 주로 가시성(Visibility) 관점에만 치우쳐 있어 포괄적인 도구 모델링에 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조 및 데이터 수집

CholecTrack20은 복강경 담낭 절제술 비디오에서 추출되었다. 데이터는 Cholec80 및 CholecT50 공개 데이터셋을 기반으로 하며, 수술의 모든 주요 단계를 포함하는 고품질 비디오를 선정하여 25 FPS에서 1 FPS로 서브샘플링하였다. 환자와 의료진의 개인정보 보호를 위해 체외(out-of-body) 프레임에 대해 익명화 처리를 수행하였다.

### 추적 정식화 (Track Formalization)

비디오 데이터셋 $D=\{S_1, S_2, \dots, S_n\}$에서 각 시퀀스 $S_i$는 바운딩 박스 $B$와 클래스 $C$로 어노테이션된다. 각 도구는 고유한 트랙 ID를 가지며, ID 재할당은 시각적 단서(클래스, 위치)와 임상적 지식(조작자의 역할, 트로카 포트 위치)을 결합하여 결정한다. 추적 문제는 시간 $t$와 $t+1$ 사이의 연관 행렬(association matrix) $A(t)$를 해결하는 것으로 정의된다:
$$A_{i,j} = 1 \text{ (i번째 도구가 j번째 도구와 연관됨), else } 0$$
최종 목표는 각 관점에 따른 도구 궤적 $T=\{T_1, T_2, \dots, T_K\}$를 얻는 것이다.

### Multi-Perspective (MP) Trajectory

본 논문은 궤적을 다음 세 가지 관점으로 구분한다:

1. **Intraoperative Trajectory (수술 내 궤적)**: 도구가 수술 중 처음 등장한 시점부터 마지막으로 사라진 시점까지를 추적한다. 폐색(occlusion), 시야 밖 이동, 재삽입 후에도 ID를 유지하는 Re-identification 능력이 요구된다.
2. **Intracorporeal Trajectory (체내 궤적)**: 도구가 신체(body)에 진입한 시점부터 트로카 포트를 통해 나가는 시점까지를 추적한다. 카메라 시야 밖에서 일어나는 진입/퇴출을 추론하여 포함하며, 수술 워크플로우 이해에 필수적이다.
3. **Visibility Trajectory (가시성 궤적)**: 도구가 카메라 시야 내에 나타난 시점부터 사라진 시점까지를 추적한다. 2초 이내의 일시적 사라짐에 대해서는 Re-ID를 통해 연결한다.

### 데이터 어노테이션 상세

- **도구 카테고리 (7종)**: cold grasper, bipolar grasper, monopolar hook, monopolar scissors, clipper, irrigator, specimen bag.
- **조작자 (4종)**: 주 수술의 왼쪽 손(MSLH), 오른쪽 손(MSRH), 보조 수술의 오른쪽 손(ASRH), NULL.
- **시각적 챌린지 (8종)**: blurring, bleeding, lens fouling, crowded scene, occlusion, smoke, specular light reflection, trocar view.
- **수술 단계 (7종)**: preparation, calot triangle dissection, gallbladder dissection, clipping & cutting, gallbladder packaging, cleaning and coagulation, gallbladder extraction.

## 📊 Results

### 도구 검출 (Tool Detection) 벤치마크

다양한 객체 검출 모델(Faster-RCNN, YOLOv7, YOLOv8, YOLOv10 등)을 테스트한 결과, **YOLOv7**이 $AP_{0.5:0.95}$ 기준으로 56.1%의 가장 높은 성능을 보였다.

- **클래스별 결과**: 'Hook'이 가장 높은 검출률을 보인 반면, 'Irrigator'와 'Specimen bag'은 경계가 불분명하거나 변형이 심해 검출이 어려웠다.
- **챌린지별 결과**: 출혈(bleeding)이나 연기(smoke) 상황에서는 상대적으로 강건했으나, 블러(blur)나 반사(reflection) 상황에서는 모든 모델의 성능이 저하되었다.

### 도구 추적 (Tool Tracking) 벤치마크

SOTA MOT 방법론(ByteTrack, Bot-SORT, SMILETrack 등)을 적용하여 분석한 결과는 다음과 같다:

- **전반적 성능**: **Bot-SORT**가 전반적으로 우수한 성능을 보였으나, 전체적인 HOTA 점수는 45% 미만으로 임상 적용을 위한 정확도에는 미치지 못했다.
- **관점별 난이도**: $\text{Visibility} < \text{Intraoperative} < \text{Intracorporeal}$ 순으로 난이도가 높았다. Visibility 추적은 시각적 단서에 의존하므로 가장 쉬웠으나, Intracorporeal 추적은 신체 진입/퇴출 시점이 명확히 보이지 않아 가장 어려웠다 (Bot-SORT 기준 HOTA 27.0%).
- **도구별 결과**: 'Grasper'는 인스턴스가 가장 많음에도 불구하고 가장 높은 추적 정확도를 보였다. 'Specimen bag'은 형태 변형과 유체 오염으로 인해 추적 성능이 낮았다.

## 🧠 Insights & Discussion

본 연구는 단순한 데이터셋 제공을 넘어, 수술 도구 추적에서 '관점(Perspective)'의 중요성을 정립하였다. 실험 결과, 기존의 SOTA 추적 모델들이 시각적 특징과 위치 정보만으로는 수술 환경의 복잡성을 극복하기 어렵다는 것이 드러났다. 특히 외형이 유사한 도구들이 많아 단순한 Appearance-based Re-ID로는 한계가 있으며, 이는 맥락 인식(context-aware) 기반의 새로운 추적 알고리즘이 필요함을 시사한다.

또한, 시각적 챌린지(연기, 출혈 등)가 추적 성능을 크게 떨어뜨리는 핵심 요인임을 확인하였으며, 이를 해결하기 위한 강건한 데이터 증강이나 도메인 특화 모델의 필요성이 제기된다. 현재의 낮은 HOTA 성능($<45\%$)은 실제 임상 환경에서 실시간 보조 시스템으로 활용되기에는 아직 갈 길이 멀다는 것을 보여주는 비판적 지표이다.

## 📌 TL;DR

본 논문은 수술 도구 추적을 위해 **Intraoperative, Intracorporeal, Visibility**라는 세 가지 관점의 궤적 정의를 도입한 새로운 데이터셋 **CholecTrack20**을 제안한다. 20개의 전체 수술 영상에 대해 도구 종류, 조작자, 수술 단계, 시각적 챌린지를 상세히 어노테이션하였으며, 벤치마크 결과 YOLOv7(검출)과 Bot-SORT(추적)가 상대적으로 우수했으나 전반적인 성능은 여전히 임상 적용 수준에 미치지 못함을 확인하였다. 이 연구는 향후 맥락 인식 기반의 강건한 수술 AI 보조 시스템 개발을 위한 기초 토대를 제공한다.
