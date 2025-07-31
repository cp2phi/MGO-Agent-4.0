import os
import json
import random
from openai import OpenAI
import time
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from typing import List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize embedding model

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-proj-xx")

# Create data directory if it doesn't exist
os.makedirs("LLM_data", exist_ok=True)

# Constants
DEFAULT_CV_COUNT = 200
DEFAULT_JD_COUNT = 40
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RISK_AVERSION = 0.2
DEFAULT_FAIL_COUNT = 0


def query_real_world_distributions() -> Tuple[Dict, Dict]:
    """
    Query LLM to get realistic distributions for CV and JD generation.
    Returns two dictionaries: cv_distributions and jd_distributions.
    """
    # Since we can't actually query the internet, we'll simulate realistic distributions
    # based on common labor market statistics

    # CV distributions
    cv_distributions = {
        "industry_distribution": {
            "Technology": 0.35,
            "Finance": 0.20,
            "Healthcare": 0.15,
            "Education": 0.10,
            "Manufacturing": 0.08,
            "Retail": 0.07,
            "Other": 0.05
        },
        "gender_distribution": {"Male": 0.52, "Female": 0.48},
        "age_distribution": {
            "20-25": 0.25,
            "26-30": 0.35,
            "31-35": 0.20,
            "36-40": 0.12,
            "41+": 0.08
        },
        "experience_distribution": {
            "0-3": 0.55,
            "3-7": 0.30,
            "7+": 0.15
        },
        "salary_expectation_distribution": {
            "20k-40k": 0.25,
            "40k-60k": 0.35,
            "60k-80k": 0.20,
            "80k-100k": 0.12,
            "100k-120k": 0.05,
            "120k+": 0.03
        },
        "location_distribution": {
            "North": 0.30,
            "South": 0.25,
            "East": 0.25,
            "West": 0.20
        },
        "job_title_distribution": {
            "Software Engineer": 0.15,
            "Data Analyst": 0.10,
            "Financial Analyst": 0.08,
            "Marketing Specialist": 0.07,
            "HR Manager": 0.06,
            "Sales Representative": 0.06,
            "Project Manager": 0.05,
            "Product Manager": 0.05,
            "UX Designer": 0.04,
            "Accountant": 0.04,
            "Nurse": 0.03,
            "Teacher": 0.03,
            "Operations Manager": 0.03,
            "Business Analyst": 0.03,
            "Customer Support": 0.03,
            "Graphic Designer": 0.02,
            "Content Writer": 0.02,
            "Research Scientist": 0.02,
            "Mechanical Engineer": 0.02,
            "Electrical Engineer": 0.02
        }
    }

    # JD distributions
    jd_distributions = {
        "hiring_personality_distribution": {
            "prefers_experience": 0.30,
            "prefers_potential": 0.25,
            "culture_fit_focused": 0.20,
            "fast_paced_environment": 0.15,
            "mission_driven": 0.10
        },
        "industry_distribution": {
            "Technology": 0.40,
            "Finance": 0.25,
            "Healthcare": 0.12,
            "Education": 0.08,
            "Manufacturing": 0.07,
            "Retail": 0.05,
            "Other": 0.03
        },
        "required_experience_distribution": {
            "0-2": 0.30,
            "2-5": 0.45,
            "5+": 0.25
        },
        "salary_range_distribution": {
            "20k-40k": 0.20,
            "40k-60k": 0.35,
            "60k-80k": 0.25,
            "80k-100k": 0.12,
            "100k-120k": 0.05,
            "120k+": 0.03
        },
        "location_distribution": {
            "North": 0.35,
            "South": 0.25,
            "East": 0.20,
            "West": 0.20
        },
        "job_title_distribution": cv_distributions["job_title_distribution"],  # Same as CV titles
    }

    # Save distributions to files
    with open("ok_data/cv_distributions.txt", "w") as f:
        json.dump(cv_distributions, f, indent=2)

    with open("ok_data/jd_distributions.txt", "w") as f:
        json.dump(jd_distributions, f, indent=2)

    return cv_distributions, jd_distributions


def sample_from_distribution(distribution: Dict) -> str:
    """Sample a value from a probability distribution dictionary."""
    rand = random.random()
    cumulative = 0.0
    for key, prob in distribution.items():
        cumulative += prob
        if rand < cumulative:
            return key
    return list(distribution.keys())[-1]  # fallback


def generate_cv_with_gpt(profile_data: Dict) -> Dict:
    """
    Generate a realistic CV using GPT based on the sampled profile data.
    Returns a complete CV dictionary.
    """
    prompt = f"""Generate a realistic CV in JSON format based on the following profile data:

    Profile Data:
    - Industry: {profile_data['industry']}
    - Gender: {profile_data['gender']}
    - Age: {profile_data['age']}
    - Experience: {profile_data['experience']} years
    - Current Job Title: {profile_data['job_title']}
    - Salary Expectation: {profile_data['salary_expectation']}
    - Location: {profile_data['location']}

    Required JSON Structure:
    {{
        "seeker_id": "SEEKER_XXXX",
        "profile": {{
            "gender": "...",
            "age": "...",
            "industry": "...",
            "skills": ["skill1", "skill2", ...],
            "experience_years": "...",
            "education": [
                {{
                    "degree": "...",
                    "field": "...",
                    "year": "..."
                }},
                ...
            ],
            "current_job_title": "...",
            "salary_expectation": "...",
            "location": "...",
            "work_history": [
                {{
                    "title": "...",
                    "company": "...",
                    "duration": "...",
                    "responsibilities": ["...", "..."]
                }},
                ...
            ]
        }},
        "summary": "A professional summary...",
        "dynamic_state": {{
            "risk_aversion": 0.2,
            "fail_count": 0
        }}
    }}

    Important Notes:
    1. Skills must be relevant to the job title and industry
    2. Education background should be realistic for the experience level
    3. Work history should be consistent with experience years
    4. Salary expectation should match the profile data
    5. All fields should be in English
    6. Make the CV realistic and professional
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system",
             "content": "You are a professional resume writer. Generate realistic CVs in JSON format based on given profile data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    cv = json.loads(response.choices[0].message.content)
    cv["seeker_id"] = profile_data["seeker_id"]  # Ensure correct ID
    cv["dynamic_state"] = {
        "risk_aversion": DEFAULT_RISK_AVERSION,
        "fail_count": DEFAULT_FAIL_COUNT
    }
    return cv


def generate_jd_with_gpt(job_data: Dict) -> Dict:
    """
    Generate a realistic job description using GPT based on the sampled job data.
    Returns a complete JD dictionary.
    """
    prompt = f"""Generate a realistic job description in JSON format based on the following job data:

    Job Data:
    - Industry: {job_data['industry']}
    - Job Title: {job_data['job_title']}
    - Required Experience: {job_data['required_experience']} years
    - Salary Range: {job_data['salary_range']}
    - Location: {job_data['location']}
    - Hiring Personality: {job_data['hiring_personality']}

    Required JSON Structure:
    {{
        "job_id": "JOB_XXXX",
        "company_name": "Company Name",
        "job_title": "...",  # Added this line
        "requirements": {{
            "required_skills": ["skill1", "skill2", ...],
            "required_experience": "...",
            "education_level": "...",
            "location": "..."
        }},
        "salary_range": "...",
        "summary": "A summary of the job...",
        "hiring_personality": "..."
    }}

    Important Notes:
    1. Required skills must be relevant to the job title and industry
    2. Education level should match the required experience
    3. Salary range should match the job data
    4. Hiring personality ({job_data['hiring_personality']}) should influence the tone of the JD:
       - prefers_experience: Emphasize years of experience and proven track record
       - prefers_potential: Focus on growth opportunities and learning ability
       - culture_fit_focused: Highlight company values and team dynamics
       - fast_paced_environment: Mention agility and adaptability requirements
       - mission_driven: Emphasize purpose and impact of the work
    5. All fields should be in English
    6. Make the JD realistic and professional
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system",
             "content": "You are a professional recruiter. Generate realistic job descriptions in JSON format based on given job data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    jd = json.loads(response.choices[0].message.content)
    jd["job_id"] = job_data["job_id"]  # Ensure correct ID
    return jd


def generate_cv(cv_id: int, distributions: Dict, seed: int = None) -> Dict:
    """Generate a single CV based on distributions."""
    if seed is not None:
        random.seed(seed + cv_id)  # Ensure reproducibility

    # Sample from distributions
    industry = sample_from_distribution(distributions["industry_distribution"])
    gender = sample_from_distribution(distributions["gender_distribution"])
    age = sample_from_distribution(distributions["age_distribution"])
    experience = sample_from_distribution(distributions["experience_distribution"])
    job_title = sample_from_distribution(distributions["job_title_distribution"])
    salary = sample_from_distribution(distributions["salary_expectation_distribution"])
    location = sample_from_distribution(distributions["location_distribution"])

    # Prepare profile data for GPT
    profile_data = {
        "seeker_id": f"SEEKER_{cv_id:04d}",
        "industry": industry,
        "gender": gender,
        "age": age,
        "experience": experience,
        "job_title": job_title,
        "salary_expectation": salary,
        "location": location
    }

    # Generate CV with GPT
    start_time = time.time()
    cv = generate_cv_with_gpt(profile_data)
    print(cv_id + ":" + time.time() - start_time)
    return cv


def generate_jd(jd_id: int, distributions: Dict, seed: int = None) -> Dict:
    """Generate a single job description based on distributions."""
    if seed is not None:
        random.seed(seed + jd_id + 10000)  # Different seed space from CVs

    # Sample from distributions
    industry = sample_from_distribution(distributions["industry_distribution"])
    job_title = sample_from_distribution(distributions["job_title_distribution"])
    req_exp = sample_from_distribution(distributions["required_experience_distribution"])
    salary = sample_from_distribution(distributions["salary_range_distribution"])
    location = sample_from_distribution(distributions["location_distribution"])
    personality = sample_from_distribution(distributions["hiring_personality_distribution"])

    # Prepare job data for GPT
    job_data = {
        "job_id": f"JOB_{jd_id:04d}",
        "industry": industry,
        "job_title": job_title,
        "required_experience": req_exp,
        "salary_range": salary,
        "location": location,
        "hiring_personality": personality
    }

    # Generate JD with GPT
    start_time = time.time()
    jd = generate_jd_with_gpt(job_data)
    print(jd_id + ":" + time.time() - start_time)

    return jd


def generate_cvs(num_cvs: int = DEFAULT_CV_COUNT, seed: int = None) -> List[Dict]:
    """Generate multiple CVs with consistent distributions."""
    cv_distributions, _ = query_real_world_distributions()
    cvs = [generate_cv(i, cv_distributions, seed) for i in range(num_cvs)]

    # Save to file
    filename = f"./data/cvs_seed{seed}.json" if seed is not None else "./data/cvs.json"
    with open(filename, "w") as f:
        json.dump(cvs, f, indent=2)

    return cvs


def generate_jds(num_jds: int = DEFAULT_JD_COUNT, seed: int = None) -> List[Dict]:
    """Generate multiple job descriptions with consistent distributions."""
    _, jd_distributions = query_real_world_distributions()
    jds = [generate_jd(i, jd_distributions, seed) for i in range(num_jds)]

    # Save to file
    filename = f"./data/jds_seed{seed}.json" if seed is not None else "./data/jds.json"
    with open(filename, "w") as f:
        json.dump(jds, f, indent=2)

    return jds


def get_combined_text_from_cv(cv: Dict) -> str:
    if 'summary' not in cv:
        cv['summary'] = "No professional summary provided"
    if 'profile' not in cv:
        cv['profile'] = {}
    if 'skills' not in cv['profile']:
        cv['profile']['skills'] = []
    if 'education' not in cv['profile']:
        cv['profile']['education'] = []
    if 'work_history' not in cv['profile']:
        cv['profile']['work_history'] = []

    """Combine relevant fields from CV into a single text for embedding."""
    profile = cv['profile']
    skills = ', '.join(profile['skills'])
    education = ', '.join([f"{edu['degree']} in {edu['field']}  from {edu['year']} " for edu in profile['education']])
    work_history = ', '.join([f"{job['title']} at {job['company']}" for job in profile['work_history']])

    return f"{cv['summary']}. Skills: {skills}. Education: {education}. Work History: {work_history}"

def get_combined_text_from_jd(jd: Dict) -> str:
    """Combine relevant fields from JD into a single text for embedding."""
    requirements = jd['requirements']
    skills = ', '.join(requirements['required_skills'])
    education = requirements.get('education_level', '')

    return f"{jd['summary']}. Required Skills: {skills}. Required Education: {education}. Hiring Personality: {jd['hiring_personality']}"


    # Compute embeddings
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_openai_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Helper function to fetch embeddings from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(data.embedding) for data in response.data]


def compute_and_save_embeddings(cvs: List[Dict], jds: List[Dict], seed: int = None):
    """Compute embeddings for all CVs and JDs and save to files."""
    # Generate combined texts
    cv_texts = [get_combined_text_from_cv(cv) for cv in cvs]
    jd_texts = [get_combined_text_from_jd(jd) for jd in jds]
    batch_size = 10  # Adjust based on your needs
    cv_embeddings = []
    jd_embeddings = []

    for i in range(0, len(cv_texts), batch_size):
        batch = cv_texts[i:i + batch_size]
        cv_embeddings.extend(get_openai_embeddings(batch))
        time.sleep(1)

    for i in range(0, len(jd_texts), batch_size):
        batch = jd_texts[i:i + batch_size]
        jd_embeddings.extend(get_openai_embeddings(batch))
        time.sleep(1)

    # Prepare data for saving
    cv_embedding_data = {
        cv['seeker_id']: embedding.tolist()
        for cv, embedding in zip(cvs, cv_embeddings)
    }

    jd_embedding_data = {
        jd['job_id']: embedding.tolist()
        for jd, embedding in zip(jds, jd_embeddings)
    }

    # Save to files
    cv_emb_filename = f"./data/cv_embeddings_seed{seed}.json" if seed is not None else "./data/cv_embeddings.json"
    jd_emb_filename = f"./data/jd_embeddings_seed{seed}.json" if seed is not None else "./data/jd_embeddings.json"

    with open(cv_emb_filename, 'w') as f:
        json.dump(cv_embedding_data, f)

    with open(jd_emb_filename, 'w') as f:
        json.dump(jd_embedding_data, f)

    return cv_embeddings, jd_embeddings


def get_top_k_similar(candidate_embedding: np.ndarray, target_embeddings: List[np.ndarray], target_ids: List[str],
                      k: int = None ) -> List[Tuple[str, float]]:
    """Get top K most similar targets based on cosine similarity."""
    similarities = cosine_similarity(
        [candidate_embedding],
        target_embeddings
    )[0]

    # Pair IDs with similarity scores
    paired = list(zip(target_ids, similarities))

    # Sort by similarity (descending)
    paired.sort(key=lambda x: x[1], reverse=True)

    # Return top K
    return paired[:k]

def generate_seeker_preference_list(cv: Dict, cv_embedding: np.ndarray, jds: List[Dict],
                                    jd_embeddings: List[np.ndarray], jd_ids: List[str],k) -> Dict:
    """Generate preference list for a single seeker using two-stage approach."""
    # Stage 1: Vector-based recall
    jd_embeddings_array = np.array(jd_embeddings)
    recalled_jds = get_top_k_similar(cv_embedding, jd_embeddings_array, jd_ids, k)
    # Sort by similarity score (descending)
    recalled_jds_sorted = sorted(recalled_jds, key=lambda x: x[1], reverse=True)

    # Prepare output format
    ranked_job_ids = [jd_id for jd_id, _ in recalled_jds_sorted]
    reasons = {
        jd_id: f"Vector similarity score: {similarity:.4f}"
        for jd_id, similarity in recalled_jds_sorted
    }

    return {
        "ranked_job_ids": ranked_job_ids,
        "reasons": reasons
    }

    ## 注释掉，不用LLM
#
#     # Prepare data for LLM ranking
#     recalled_jd_data = []
#     for jd_id, similarity in recalled_jds:
#         jd = next(jd for jd in jds if jd['job_id'] == jd_id)
#         recalled_jd_data.append({
#             'job_id': jd_id,
#             'job_title': jd.get('job_title', ''),
#             'company_name': jd.get('company_name', ''),
#             'summary': jd.get('summary', ''),
#             'requirements': jd.get('requirements', {})
#         })
#
#     # Stage 2: LLM-based reranking with optimized prompt
#     prompt = f"""You are the candidate with this profile. Given these job opportunities, rank them based on what would be best for YOUR career.
#
# Your Profile:
# - Current Job Title: {cv['profile']['current_job_title']}
# - Skills: {', '.join(cv['profile']['skills'])}
# - Experience: {cv['profile']['experience_years']} years
# - Education: {', '.join([f"{edu['degree']} in {edu['field']}" for edu in cv['profile']['education']])}
# - Salary Expectation: {cv['profile']['salary_expectation']}
# - Location: {cv['profile']['location']}
# - Career Summary: {cv['summary']}
#
# Job Opportunities to Rank:
# {json.dumps(recalled_jd_data, indent=2)}
#
# Ranking Criteria (in order of importance):
# 1. Title Match: How well the job title aligns with your current role and career goals
# 2. Skills Match: Percentage of your skills that match the job requirements
# 3. Experience Fit: Whether the required experience matches your background
# 4. Career Growth: Potential for advancement and skill development
# 5. Compensation: How well the salary range matches your expectations
# 6. Location: Convenience and suitability of the job location
#
# Output Format (JSON):
# {{
#     "ranked_job_ids": ["JOB_1234", "JOB_5678", ...],
#     "reasons": {{
#         "JOB_1234": "Excellent title match (Data Scientist) and 85% skills alignment",
#         "JOB_5678": "Good growth opportunity but requires relocation",
#         ...
#     }}
# }}
#
# Important:
# - Return exactly {len(recalled_jd_data)} ranked jobs
# - Be specific in your reasoning for each ranking
# - Consider yourself as the candidate making this decision
# """
#     start_time = time.time()
#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": "You are a professional making career decisions. Carefully rank job opportunities from the candidate's perspective."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0,
#         response_format={"type": "json_object"}
#     )
#     print(time.time() - start_time)
#     return json.loads(response.choices[0].message.content)

def generate_jd_preference_list(jd: Dict, jd_embedding: np.ndarray, cvs: List[Dict], cv_embeddings: List[np.ndarray],
                                cv_ids: List[str],k) -> Dict:
    """Generate preference list for a single job using two-stage approach."""
    # Stage 1: Vector-based recall
    cv_embeddings_array = np.array(cv_embeddings)
    recalled_cvs = get_top_k_similar(jd_embedding, cv_embeddings_array, cv_ids,int(k*2))
    # Sort by similarity score (descending)
    random.shuffle(recalled_cvs)
    recalled_cvs = recalled_cvs[:int(k/2)]
    # recalled_cvs_sorted = sorted(recalled_cvs, key=lambda x: x[1], reverse=True)

    # Prepare output format
    ranked_cv_ids = [cv_id for cv_id, _ in recalled_cvs]
    reasons = {
        jd_id: f"Vector similarity score: {similarity:.4f}"
        for jd_id, similarity in recalled_cvs
    }

    return {
        "ranked_candidate_ids": ranked_cv_ids,
        "reasons": reasons
    }
#     # Prepare data for LLM ranking
#     recalled_cv_data = []
#     for cv_id, similarity in recalled_cvs:
#         cv = next(cv for cv in cvs if cv['seeker_id'] == cv_id)
#         recalled_cv_data.append({
#             'seeker_id': cv_id,
#             'current_job_title': cv['profile']['current_job_title'],
#             'skills': cv['profile']['skills'],
#             'experience': cv['profile']['experience_years'],
#             'education': [f"{edu['degree']} in {edu['field']}" for edu in cv['profile']['education']],
#             'salary_expectation': cv['profile']['salary_expectation'],
#             'location': cv['profile']['location'],
#             'summary': cv['summary']
#         })
#
#     # Stage 2: LLM-based reranking with optimized prompt
#     prompt = f"""As the HR manager for {jd.get('company_name', 'the company')}, rank these candidates for the {jd.get('job_title', '')} position based on our specific needs.
#
# Job Requirements:
# - Title: {jd.get('job_title', '')}
# - Key Skills Needed: {', '.join(jd['requirements'].get('required_skills', []))}
# - Required Experience: {jd['requirements'].get('required_experience', '')}
# - Education Level: {jd['requirements'].get('education_level', '')}
# - Hiring Personality: {jd['hiring_personality']}
# - Job Summary: {jd.get('summary', '')}
#
# Candidate Profiles:
# {json.dumps(recalled_cv_data, indent=2)}
#
# Ranking Criteria (in order of importance):
# 1. Skills Match: Percentage of required skills the candidate possesses
# 2. Experience Fit: How closely their experience matches requirements
# 3. Title Relevance: Relevance of their current/most recent position
# 4. Cultural Fit: Alignment with our {jd['hiring_personality']} hiring personality
# 5. Education: Match with required education level
# 6. Location: Willingness/ability to work in our location
#
# Output Format (JSON):
# {{
#     "ranked_candidate_ids": ["SEEKER_1234", "SEEKER_5678", ...],
#     "reasons": {{
#         "SEEKER_1234": "90% skills match and perfect experience fit",
#         "SEEKER_5678": "Strong technical skills but needs more management experience",
#         ...
#     }}
# }}
#
# Important:
# - Return exactly {len(recalled_cv_data)} ranked candidates
# - Be specific in your reasoning for each ranking
# - Consider our company's specific needs and culture
# """
#     start_time = time.time()
#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": "You are an experienced HR professional making hiring decisions. Carefully rank candidates from the company's perspective."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0,
#         response_format={"type": "json_object"}
#     )
#     print(time.time() - start_time)
#
#     return json.loads(response.choices[0].message.content)

def generate_all_preference_lists(cvs: List[Dict], jds: List[Dict], seed: int = None, k: int = None):
    """Generate and save all preference lists for CVs and JDs."""
    # Compute or load embeddings
    cv_emb_filename = f"./data/cv_embeddings_seed{seed}.json" if seed is not None else "./data/cv_embeddings.json"
    jd_emb_filename = f"./data/jd_embeddings_seed{seed}.json" if seed is not None else "./data/jd_embeddings.json"

    if os.path.exists(cv_emb_filename) and os.path.exists(jd_emb_filename):
        # Load existing embeddings
        with open(cv_emb_filename, 'r') as f:
            cv_embeddings_data = json.load(f)
        with open(jd_emb_filename, 'r') as f:
            jd_embeddings_data = json.load(f)

        # Convert back to numpy arrays in original order
        cv_embeddings = [np.array(cv_embeddings_data[cv['seeker_id']]) for cv in cvs]
        jd_embeddings = [np.array(jd_embeddings_data[jd['job_id']]) for jd in jds]
    else:
        # Compute new embeddings
        print("ERROR: embedding file empty")

    # Get IDs for reference
    cv_ids = [cv['seeker_id'] for cv in cvs]
    jd_ids = [jd['job_id'] for jd in jds]

    # Generate seeker preference lists
    seeker_preferences = {}
    for cv, cv_emb in zip(cvs, cv_embeddings):
        pref_list = generate_seeker_preference_list(cv, cv_emb, jds, jd_embeddings, jd_ids,k)
        seeker_preferences[cv['seeker_id']] = pref_list

    # Generate JD preference lists
    jd_preferences = {}
    for jd, jd_emb in zip(jds, jd_embeddings):
        pref_list = generate_jd_preference_list(jd, jd_emb, cvs, cv_embeddings, cv_ids,k)
        jd_preferences[jd['job_id']] = pref_list

    # Save preference lists
    seeker_pref_filename = f"./data/seeker_preferences_seed{seed}.json" if seed is not None else "./data/seeker_preferences.json"
    jd_pref_filename = f"./data/jd_preferences_seed{seed}.json" if seed is not None else "./data/jd_preferences.json"

    with open(seeker_pref_filename, 'w') as f:
        json.dump(seeker_preferences, f, indent=2)

    with open(jd_pref_filename, 'w') as f:
        json.dump(jd_preferences, f, indent=2)

    return seeker_preferences, jd_preferences

if __name__ == "__main__":
    # Example usage
    seed = 42  # Change this for different random generations
    #
    # # 1. Generate 25 CVs with seed=42
    # cvs = generate_cvs(50, seed)
    # print(f"Generated {len(cvs)} CVs")
    #
    # # 2. Generate 10 JDs with seed=42
    # jds = generate_jds(20, seed)
    # print(f"Generated {len(jds)} JDs")

    # cv_embeddings, jd_embeddings = compute_and_save_embeddings(cvs, jds, seed)

    #######################
    cvs_filename = f"ok_data/cvs_seed{seed}.json"
    jds_filename = f"ok_data/jds_seed{seed}.json"

    with open(cvs_filename, 'r', encoding='utf-8') as f:
        cvs = json.load(f)
    print(json.dumps(cvs[0], indent=2))

    with open(jds_filename, 'r', encoding='utf-8') as f:
        jds = json.load(f)
    print(json.dumps(jds[0], indent=2))
#######################

    # 3. Generate all preference lists
    seeker_prefs, jd_prefs = generate_all_preference_lists(cvs, jds, seed, k=10)

#######################
    # cvs_filename = f"./data/cvs_seed{seed}.json"
    # jds_filename = f"./data/jds_seed{seed}.json"
    #
    # jd_prefs_filename = f"./data/jd_preferences_seed{seed}.json"
    # seeker_prefs_filename= f"./data/seeker_preferences_seed{seed}.json"
    #
    # with open(jd_prefs_filename, 'r', encoding='utf-8') as f:
    #     jd_prefs = json.load(f)
    # example_jd = jds[0]['job_id']
    # print(json.dumps(jd_prefs[example_jd], indent=2))
    #
    # with open(seeker_prefs_filename, 'r', encoding='utf-8') as f:
    #     seeker_prefs = json.load(f)
    # example_seeker = cvs[0]['seeker_id']
    # print(json.dumps(seeker_prefs[example_seeker], indent=2))
########################

