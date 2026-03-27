import streamlit as st
import pickle
import math
import re
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import plotly.graph_objects as go

st.set_page_config(page_title="نظام البحث العربي", page_icon="🔍", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    * { font-family: 'Cairo', sans-serif; direction: rtl; text-align: right; }
    .stButton > button { width: 100%; background-color: #4CAF50; color: white; }
    .highlight { background-color: #fffacd; padding: 2px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

class TextPreprocessor:
    def __init__(self, stopwords_file='arabic_stop_words (2).txt'):
        self.stopwords = set()
        if stopwords_file and os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    self.stopwords = {self.normalize(w.strip()) for w in f if w.strip()}
            except:
                pass

    def normalize(self, text):
        if not text:
            return ""
        text = re.sub(r'[\u064B-\u065F]', '', text)
        text = re.sub(r'[آأإا]', 'ا', text)
        text = re.sub(r'[ة]', 'ه', text)
        text = re.sub(r'[ى]', 'ي', text)
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess(self, text, stem=True):
        if not text:
            return []
        text = self.normalize(text)
        tokens = [t for t in re.findall(r'[\u0600-\u06FF]+', text) 
                 if t not in self.stopwords and len(t) > 1]
        
        if stem:
            stemmed = []
            for token in tokens:
                token = re.sub(r'^ال', '', token)
                while token and token[0] in 'وفبلس':
                    token = token[1:]
                if len(token) > 4:
                    token = re.sub(r'(ات|ون|ين)$', '', token)
                token = re.sub(r'(كم|نا|كما|هما|هم|هن|ها|ه|ك|ي|ت)$', '', token) if len(token) > 4 else re.sub(r'(كم|نا|هم|هن|ها|ه|ك|ي|ت)$', '', token)
                if token:  # Only add non-empty tokens
                    stemmed.append(token)
            return stemmed
        return tokens

class Indexer(TextPreprocessor):
    def __init__(self):
        super().__init__()
        self.index = {}
        self.doc_texts = {}
        self.sentiments = {}
        self.doc_count = 0
        self.doc_vectors = {}
        self.doc_norms = {}

    def add_document(self, doc_id, text, sentiment):
        tokens = self.preprocess(text)
        if not tokens:
            return False
        self.doc_texts[doc_id] = text
        self.sentiments[doc_id] = sentiment
        self.doc_count += 1
        seen = set()
        for pos, term in enumerate(tokens, 1):
            if term not in self.index:
                self.index[term] = {"df": 0, "docs": {}}
            if doc_id not in self.index[term]["docs"]:
                self.index[term]["docs"][doc_id] = []
            self.index[term]["docs"][doc_id].append(pos)
            seen.add(term)
        for term in seen:
            self.index[term]["df"] += 1
        return True

    def load_xml(self, xml_file):
        if not os.path.exists(xml_file):
            return 0
        added = 0
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            elements = root.findall('.//document')
            if not elements:
                elements = root.findall('.//doc')
                st.info("⚠️ لم يتم العثور على عناصر 'document'، استخدام عناصر 'doc' بدلاً من ذلك")
            st.info(f"✅ تم العثور على {len(elements)} عنصر في الملف")
            processed_ids = set()  # لتجنب التكرار
            for elem in elements:
                xml_id_attr = elem.get('id')
                if xml_id_attr:
                    try:
                        doc_id = int(xml_id_attr)
                    except:
                        doc_id = added + 1
                else:
                    doc_id = added + 1
                if doc_id in processed_ids:
                    continue
                text_elem = elem.find('text')
                if text_elem is not None and text_elem.text:
                    text = text_elem.text.strip()
                else:
                    text = re.sub(r'\s+', ' ', ' '.join(t.strip() for t in elem.itertext() if t and t.strip())).strip()
                if len(text) < 3:
                    continue
                sentiment = self._extract_sentiment(elem)
                if self.add_document(doc_id, text, sentiment if sentiment is not None else 1):
                    processed_ids.add(doc_id)
                    added += 1
            st.success(f"✅ تمت إضافة {added} وثيقة إلى الفهرس")
            return added
        except Exception as e:
            st.error(f"❌ خطأ في تحميل XML: {e}")
            import traceback
            st.error(traceback.format_exc())
            return 0

    def analyze_xml_structure(self, xml_file):
        """Analyze XML structure to understand the issue"""
        if not os.path.exists(xml_file):
            return "File not found"
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            analysis = []
            analysis.append(f"Root tag: {root.tag}")
            all_elements = list(root.iter())
            analysis.append(f"Total elements: {len(all_elements)}")
            tag_counts = {}
            for elem in root.iter():
                tag = elem.tag
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            analysis.append("\nTag counts:")
            for tag, count in sorted(tag_counts.items()):
                analysis.append(f"  {tag}: {count}")
            doc_elements = root.findall('.//document')
            analysis.append(f"\n'document' elements: {len(doc_elements)}")
            doc_elements = root.findall('.//doc')
            analysis.append(f"'doc' elements: {len(doc_elements)}")
            if doc_elements:
                sample = doc_elements[0]
                analysis.append(f"\nSample document structure:")
                analysis.append(f"  Tag: {sample.tag}")
                analysis.append(f"  ID attribute: {sample.get('id')}")
                analysis.append(f"  Has text child: {sample.find('text') is not None}")
                if sample.find('text') is not None:
                    text_elem = sample.find('text')
                    text_content = text_elem.text[:100] + "..." if text_elem.text and len(text_elem.text) > 100 else (text_elem.text or "Empty")
                    analysis.append(f"  Text preview: {text_content}")
            return "\n".join(analysis)
        except Exception as e:
            return f"Error analyzing XML: {str(e)}"
        
    def _extract_sentiment(self, elem):
        for attr in ['sentiment', 'polarity', 'label', 'class', 'emotion']:
            val = elem.get(attr)
            if val:
                return self._map_sentiment(val)
        for tag in ['sentiment', 'polarity', 'label', 'class', 'emotion']:
            se = elem.find(tag)
            if se is not None and se.text:
                return self._map_sentiment(se.text)
        return None

    def _map_sentiment(self, val):
        if not val:
            return 1
        v = str(val).strip().lower()
        if any(t in v for t in ['neg', 'سلبي', 'سلبية', '-1']):
            return 0
        if any(t in v for t in ['pos', 'إيجابي', 'positive', 'ايجابية', 'سعيد', '1', '2']):
            return 2
        return 1

    def compute_tf_weight(self, freq):
        """Compute TF weight using logarithmic scaling"""
        if freq == 0:
            return 0
        return 1 + math.log10(freq)
    
    def compute_idf(self, term):
        """Compute IDF weight"""
        if term not in self.index:
            return 0
        N = self.doc_count
        df = self.index[term]['df']
        if df == 0:
            return 0
        return math.log10(N / df)
    
    def compute_all_weights(self):
        """Compute TF-IDF weights for all terms in all documents"""
        print(f"Computing TF-IDF weights for {self.doc_count} documents...")
        
        for doc_id in self.doc_texts.keys():
            vec = {}
            for term, meta in self.index.items():
                if doc_id in meta['docs']:
                    freq = len(meta['docs'][doc_id])
                    tf_weight = self.compute_tf_weight(freq)
                    idf = self.compute_idf(term)
                    weight = tf_weight * idf
                    if weight > 0:
                        vec[term] = weight
            self.doc_vectors[doc_id] = vec
            self.doc_norms[doc_id] = math.sqrt(sum(w ** 2 for w in vec.values()))
        print(f"✓ Computed weights for {len(self.doc_vectors)} documents")

    def save_index(self, output_dir="index_data"):
        os.makedirs(output_dir, exist_ok=True)
        self.compute_all_weights()
        with open(f"{output_dir}/index.pkl", "wb") as f:
            pickle.dump({
                'index': self.index, 
                'doc_texts': self.doc_texts, 
                'sentiments': self.sentiments, 
                'doc_count': self.doc_count,
                'doc_vectors': self.doc_vectors,
                'doc_norms': self.doc_norms
            }, f)
        self.save_index_txt(f"{output_dir}/index.txt")

    def save_index_txt(self, output_file="index.txt"):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\nفهرس نظام البحث العربي\n" + "=" * 80 + "\n\n")
                f.write(f"إجمالي الوثائق: {self.doc_count}\nإجمالي المصطلحات: {len(self.index)}\n\n" + "=" * 80 + "\n\n")
                for term in sorted(self.index.keys()):
                    td = self.index[term]
                    f.write(f"المصطلح: {term}\n  تكرار الوثيقة (DF): {td['df']}\n  عدد الوثائق: {len(td['docs'])}\n")
                    for doc_id in sorted(td['docs'].keys()):
                        f.write(f"    الوثيقة {doc_id}: المواضع {td['docs'][doc_id]}\n")
                    f.write("\n")
                f.write("=" * 80 + "\nنهاية الفهرس\n" + "=" * 80 + "\n")
            return True
        except Exception as e:
            st.error(f"خطأ: {e}")
            return False

class SearchSystem(TextPreprocessor):
    def __init__(self):
        super().__init__()
        self.index = None
        self.doc_texts = None
        self.sentiments = None
        self.doc_count = 0
        self.doc_vectors = {}
        self.doc_norms = {}
        self.synonyms = {self.normalize(k): [self.normalize(v) for v in vals] 
                        for k, vals in {'صحه': ['طب', 'علاج'], 'تعليم': ['دراسه', 'مدرسه'],
                                       'اقتصاد': ['مال', 'تجاره'], 'رياضه': ['لاعب', 'مباراه'],
                                       'مشكله': ['صعوبه', 'ازمه'], 'خدمه': ['عمل', 'وظيفه']}.items()}

    def load(self, index_dir="index_data"):
        try:
            with open(f"{index_dir}/index.pkl", "rb") as f:
                data = pickle.load(f)
                self.index = data['index']
                self.doc_texts = data['doc_texts']
                self.sentiments = data['sentiments']
                self.doc_count = data['doc_count']
                self.doc_vectors = data.get('doc_vectors', {})
                self.doc_norms = data.get('doc_norms', {})
            if not self.doc_vectors:
                return False, "⚠️ الأوزان المحسوبة مسبقاً غير موجودة. يرجى إعادة بناء الفهرس."
            return True, f"تم تحميل {self.doc_count} وثيقة"
        except Exception as e:
            return False, f"خطأ: {str(e)}"

    def query_vector(self, terms):
        """Compute query vector using TF-IDF"""
        counts = defaultdict(int)
        for t in terms: 
            counts[t] += 1
        vec = {}
        for t, c in counts.items():
            if t in self.index:
                tf = 1 + math.log10(c)
                idf = math.log10(self.doc_count / self.index[t]['df'])
                vec[t] = tf * idf
        return vec

    def cosine_similarity(self, qvec, doc_id):
        if doc_id not in self.doc_vectors or not qvec: 
            return 0.0
        dvec = self.doc_vectors[doc_id]
        dot = sum(qvec.get(t, 0) * dvec.get(t, 0) for t in qvec if t in dvec)
        qnorm = math.sqrt(sum(v*v for v in qvec.values()))
        dnorm = self.doc_norms.get(doc_id, 0)
        
        return dot / (qnorm * dnorm) if qnorm and dnorm else 0.0

    def rank_docs(self, doc_ids, query_terms, top_k=150):
        if not doc_ids: 
            return []
        qvec = self.query_vector(query_terms)
        results = [(doc_id, self.cosine_similarity(qvec, doc_id)) 
                   for doc_id in doc_ids]
        results = [(d, s) for d, s in results if s > 0]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def search(self, query_text, expand=True, sentiment_filter=None):
        if not query_text.strip() or self.index is None:
            return []
        m = re.match(r'^#(\d+)\(([^,]+),\s*([^)]+)\)$', query_text)
        if m:
            dist, t1, t2 = int(m.group(1)), self.preprocess(m.group(2))[0] if self.preprocess(m.group(2)) else None, self.preprocess(m.group(3))[0] if self.preprocess(m.group(3)) else None
            if t1 and t2:
                docs = self._proximity(t1, t2, dist)
                ranked = self.rank_docs(docs, [t1, t2])
                return [(d, s) for d, s in ranked if sentiment_filter is None or self.sentiments.get(d) == sentiment_filter][:150]
            return []
        if query_text.startswith('"') and query_text.endswith('"'):
            terms = self.preprocess(query_text[1:-1])
            if terms:
                docs = self._phrase(terms)
                ranked = self.rank_docs(docs, terms)
                return [(d, s) for d, s in ranked if sentiment_filter is None or self.sentiments.get(d) == sentiment_filter][:150]
            return []

        if re.search(r'\b(AND|OR|NOT)\b|\(|\)', query_text, re.I) or re.search(r'\b(أو|لا|ليس|و)\b', query_text, re.I):
            docs = self._boolean(query_text)
            terms = self.preprocess(query_text)
            ranked = self.rank_docs(docs, terms)
            return [(d, s) for d, s in ranked if sentiment_filter is None or self.sentiments.get(d) == sentiment_filter][:150]
        terms = self.preprocess(query_text)
        if not terms:
            return []
        if expand:
            for term in list(terms):
                if term in self.synonyms:
                    terms.extend([syn for syn in self.synonyms[term] if syn not in terms and syn in self.index])
        candidates = set()
        for term in terms:
            if term in self.index:
                candidates.update(self.index[term]["docs"].keys())
        if not candidates:
            return []
        ranked = self.rank_docs(candidates, terms)
        return [(d, s) for d, s in ranked if sentiment_filter is None or self.sentiments.get(d) == sentiment_filter][:150]

    def _proximity(self, t1, t2, dist):
        if t1 not in self.index or t2 not in self.index:
            return set()
        docs = set(self.index[t1]['docs'].keys()) & set(self.index[t2]['docs'].keys())
        res = set()
        for d in docs:
            p1, p2 = self.index[t1]['docs'][d], self.index[t2]['docs'][d]
            i = j = 0
            while i < len(p1) and j < len(p2):
                if abs(p1[i] - p2[j]) <= dist:
                    res.add(d)
                    break
                if p1[i] < p2[j]:
                    i += 1
                else:
                    j += 1
        return res

    def _phrase(self, terms):
        if not terms:
            return set()
        candidates = set(self.index.get(terms[0], {}).get('docs', {}).keys())
        for term in terms[1:]:
            candidates &= set(self.index.get(term, {}).get('docs', {}).keys())
        res = set()
        for d in candidates:
            for start in self.index[terms[0]]['docs'][d]:
                if all((start + i) in self.index[terms[i]]['docs'][d] for i in range(1, len(terms))):
                    res.add(d)
                    break
        return res

    def _boolean(self, query):
        qt = re.sub(r'\b(أو|و|لا|ليس)\b', lambda m: {'أو': 'OR', 'و': 'AND', 'لا': 'NOT', 'ليس': 'NOT'}[m.group()], query, flags=re.I)
        tokens = [t for t in re.findall(r'\(|\)|\b(?:AND|OR|NOT)\b|[^\s()]+', qt, re.I) if t.strip()]
        prec = {'NOT': 3, 'AND': 2, 'OR': 1}
        output, stack = [], []
        for tok in tokens:
            up = tok.upper()
            if tok == '(':
                stack.append(tok)
            elif tok == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack:
                    stack.pop()
            elif up in prec:
                while stack and stack[-1] != '(' and (prec.get(stack[-1].upper(), 0) > prec[up] or (prec.get(stack[-1].upper(), 0) == prec[up] and up != 'NOT')):
                    output.append(stack.pop())
                stack.append(up)
            else:
                t = self.preprocess(tok)
                output.append(('TERM', t[0] if t else None))
        
        while stack:
            output.append(stack.pop())
        
        eval_stack = []
        universe = set(self.doc_texts.keys())
        for item in output:
            if isinstance(item, tuple) and item[0] == 'TERM':
                eval_stack.append(set(self.index.get(item[1], {}).get('docs', {}).keys()) if item[1] else set())
            elif item.upper() == 'NOT':
                eval_stack.append(universe - (eval_stack.pop() if eval_stack else set()))
            elif item.upper() in ('AND', 'OR'):
                if len(eval_stack) >= 2:
                    b, a = eval_stack.pop(), eval_stack.pop()
                    eval_stack.append(a & b if item.upper() == 'AND' else a | b)
        return eval_stack[0] if eval_stack else set()

    def highlight_terms(self, text, terms):
        for term in terms:
            if len(term) >= 2:
                text = re.sub(rf'\b{re.escape(term)}\b', r'<span class="highlight">\g<0></span>', text, flags=re.I)
        return text

    def analyze_sentiment_distribution(self, results):
        if not results:
            return None
        counts = Counter(self.sentiments.get(d, 1) for d, _ in results)
        total = len(results)
        return {k: {'count': counts[i], 'percentage': counts[i]/total*100 if total else 0} 
                for k, i in [('negative', 0), ('neutral', 1), ('positive', 2)]}

    def run_batch_queries(self, queries_file='queries.txt'):
        try:
            if self.index is None:
                return False, "الفهرس غير محمل"
            
            results_info = {
                'boolean_queries': 0, 
                'ranked_queries': 0, 
                'boolean_results': 0, 
                'ranked_results': 0
            }
            query_lines = []
            if os.path.exists(queries_file):
                with open(queries_file, 'r', encoding='utf-8') as f:
                    query_lines = [l.strip() for l in f if l.strip()]
            else:
                return False, f"❌ ملف {queries_file} غير موجود. يرجى إنشاء الملف وإضافة الاستعلامات."
            
            if not query_lines:
                return False, f"⚠️ ملف {queries_file} فارغ. يرجى إضافة استعلامات."
            with open('results.boolean.txt', 'w', encoding='utf-8') as f:
                for qnum, line in enumerate(query_lines, 1):
                    try:
                        is_boolean = (
                            re.search(r'\b(AND|OR|NOT)\b', line, re.I) or 
                            re.search(r'\b(أو|لا|ليس|و)\b', line, re.I) or
                            '(' in line or ')' in line
                        )
                        if is_boolean:
                            docs = self._boolean(line)
                        else:
                            terms = self.preprocess(line)
                            if terms:
                                sets = []
                                for t in terms:
                                    if t in self.index:
                                        sets.append(set(self.index[t]['docs'].keys()))
                                if sets:
                                    docs = set.intersection(*sets)
                                else:
                                    docs = set()
                            else:
                                docs = set()
                        if docs:
                            for doc_id in sorted(docs):
                                f.write(f"{qnum},{doc_id}\n")
                                results_info['boolean_results'] += 1
                        results_info['boolean_queries'] += 1
                    except Exception as e:
                        print(f"Error processing boolean query {qnum}: {line} - {e}")
                        results_info['boolean_queries'] += 1
            with open('results.ranked.txt', 'w', encoding='utf-8') as f:
                for qnum, line in enumerate(query_lines, 1):
                    try:
                        results = self.search(line, expand=True, sentiment_filter=None)
                        results = results[:150]
                        for doc_id, score in results:
                            f.write(f"{qnum},{doc_id},{score:.4f}\n")
                            results_info['ranked_results'] += 1
                        results_info['ranked_queries'] += 1
                    except Exception as e:
                        print(f"Error processing ranked query {qnum}: {line} - {e}")
                        results_info['ranked_queries'] += 1
            return True, {
                'boolean': 'results.boolean.txt', 
                'ranked': 'results.ranked.txt',
                'stats': results_info
            }
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return False, error_msg

    def debug_index_status(self):
        """Debug method to check index status"""
        print(f"Total documents: {self.doc_count}")
        print(f"Total terms in index: {len(self.index)}")
        print(f"Documents with precomputed vectors: {len(self.doc_vectors)}")
        
        # Show sample terms
        if self.index:
            sample_terms = list(self.index.keys())[:5]
            print(f"\nSample terms: {sample_terms}")
            for term in sample_terms:
                print(f"  {term}: {len(self.index[term]['docs'])} documents")
        test_query = "مشكلة"
        print(f"\nTest search for '{test_query}':")
        results = self.search(test_query, expand=False)
        print(f"  Found {len(results)} results")
        if results:
            print(f"  Top result: doc {results[0][0]}, score {results[0][1]:.4f}")

def main():
    if 'search_system' not in st.session_state:
        st.session_state.search_system = SearchSystem()
        st.session_state.system_loaded = False
        st.session_state.current_query = ""
    
    st.title("🔍 نظام البحث العربي بتحليل المشاعر")
    with st.sidebar:
        st.title("⚙️ الإعدادات")
        xml_file = "twifil.xml"
        if os.path.exists(xml_file):
            st.success("✅ تم العثور على ملف البيانات")
            if st.button("🔍 تحليل هيكل XML"):
                with st.spinner("جارٍ تحليل هيكل الملف..."):
                    indexer = Indexer()
                    analysis = indexer.analyze_xml_structure(xml_file)
                    st.text_area("نتيجة التحليل", analysis, height=400)
            if st.button("🚀 إنشاء الفهرس", type="primary"):
                with st.spinner("جاري إنشاء الفهرس..."):
                    indexer = Indexer()
                    num_docs = indexer.load_xml(xml_file)
                    if num_docs > 0:
                        indexer.save_index()
                        success, message = st.session_state.search_system.load()
                        if success:
                            st.session_state.system_loaded = True
                            st.success(f"✅ تم فهرسة {num_docs} وثيقة")
                            st.info("✓ تم حساب وحفظ أوزان TF-IDF")
                            if os.path.exists('index_data/index.txt'):
                                with open('index_data/index.txt', 'r', encoding='utf-8') as f:
                                    st.download_button("⬇️ تحميل ملف الفهرس (index.txt)", f.read(), 
                                                     file_name="index.txt", mime="text/plain")
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("❌ لم يتم تحميل أي وثائق")
        else:
            st.error(f"❌ ملف {xml_file} غير موجود")
        st.divider()
        if st.session_state.system_loaded:
            sys = st.session_state.search_system
            st.success("✅ النظام جاهز")
            st.info(f"الوثائق: {sys.doc_count}")
            st.info(f"الأوزان المحسوبة: {len(sys.doc_vectors)}")
            if os.path.exists('index_data/index.txt'):
                with open('index_data/index.txt', 'r', encoding='utf-8') as f:
                    st.download_button("📥 تحميل ملف الفهرس (TXT)", f.read(), 
                                     file_name="index.txt", mime="text/plain")
            st.divider()
            st.subheader("📝 ملف الاستعلامات")
            if os.path.exists('queries.txt'):
                with open('queries.txt', 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        with st.expander("👁️ معاينة ملف الاستعلامات الحالي"):
                            st.code(content, language='text')
            if os.path.exists('queries.txt'):
                with open('queries.txt', 'r', encoding='utf-8') as f:
                    st.download_button("⬇️ تحميل ملف الاستعلامات الحالي", f.read(), 
                                     file_name="queries.txt", mime="text/plain")
            uploaded_file = st.file_uploader("رفع ملف استعلامات جديد", type=['txt'])
            if uploaded_file is not None:
                with open('queries.txt', 'wb') as f:
                    f.write(uploaded_file.getvalue())
                st.success(f"✅ تم رفع ملف {uploaded_file.name} واستبدال queries.txt")
                st.rerun()
            st.divider()
            if st.button("🗂️ توليد ملفات النتائج"):
                with st.spinner("جارٍ تنفيذ الاستعلامات..."):
                    ok, info = st.session_state.search_system.run_batch_queries('queries.txt')
                    if ok:
                        st.success("✅ تم إنشاء ملفات النتائج")
                        if 'stats' in info:
                            stats = info['stats']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("عدد الاستعلامات", stats['boolean_queries'])
                                st.metric("نتائج Boolean", stats['boolean_results'])
                            with col2:
                                st.metric("نفس الاستعلامات", stats['ranked_queries'])
                                st.metric("نتائج Ranked", stats['ranked_results'])
                        for k in ['boolean', 'ranked']:
                            if k in info and os.path.exists(info[k]):
                                with open(info[k], 'r', encoding='utf-8') as f:
                                    label = "منطقية" if k == 'boolean' else "مرتبة"
                                    st.download_button(f"⬇️ تحميل نتائج {label}", f.read(), 
                                                     file_name=os.path.basename(info[k]), mime='text/plain')
                    else:
                        st.error(f"❌ خطأ: {info}")
        else:
            st.warning("⚠️ النظام غير محمل")
        
        st.divider()
        st.subheader("🔍 خيارات البحث")
        expand_query = st.checkbox("توسيع الاستعلام", value=True)
        show_sentiment = st.checkbox("عرض المشاعر", value=True)
        st.subheader("🎭 فلترة")
        sentiment_filter = st.selectbox("تصفية النتائج", 
                                       ["جميع المشاعر", "إيجابي فقط", "سلبي فقط", "محايد فقط"])
        st.divider()
        st.subheader("💡 أمثلة")
        for example in ["مشكلة", "تعليم", "اقتصاد", "خدمة"]:
            if st.button(example):
                st.session_state.current_query = example
                st.session_state.auto_search = True
    if not st.session_state.system_loaded:
        st.warning("## 🎯 مرحباً بك\n\n1. تأكد من وجود ملف `twifil.xml`\n2. اضغط 'إنشاء الفهرس'\n3. ابدأ البحث")
        return
    sys = st.session_state.search_system
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("الوثائق", sys.doc_count)
    with col2:
        st.metric("المفردات", len(sys.index))
    with col3:
        st.metric("الحالة", "🟢")
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("أدخل كلمات البحث", value=st.session_state.current_query, 
                             placeholder="مثال: مشكلة في التعليم")
    with col2:
        search_button = st.button("🔍 بحث", type="primary")
    sentiment_map = {
        "جميع المشاعر": None,
        "إيجابي فقط": 2,
        "سلبي فقط": 0,
        "محايد فقط": 1
    }
    selected_sentiment = sentiment_map[sentiment_filter]
    if search_button or (hasattr(st.session_state, 'auto_search') and st.session_state.auto_search):
        if hasattr(st.session_state, 'auto_search'):
            st.session_state.auto_search = False
        if query.strip():
            with st.spinner("جاري البحث..."):
                results = sys.search(query, expand=expand_query, sentiment_filter=selected_sentiment)
                if results:
                    st.success(f"تم العثور على {len(results)} نتيجة")
                    if show_sentiment:
                        sentiment_dist = sys.analyze_sentiment_distribution(results)
                        if sentiment_dist:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("😊 إيجابي", f"{sentiment_dist['positive']['count']} ({sentiment_dist['positive']['percentage']:.1f}%)")
                            with col2:
                                st.metric("😐 محايد", f"{sentiment_dist['neutral']['count']} ({sentiment_dist['neutral']['percentage']:.1f}%)")
                            with col3:
                                st.metric("😞 سلبي", f"{sentiment_dist['negative']['count']} ({sentiment_dist['negative']['percentage']:.1f}%)")
                            fig = go.Figure(data=[go.Pie(
                                labels=['إيجابي', 'محايد', 'سلبي'],
                                values=[sentiment_dist['positive']['count'], 
                                       sentiment_dist['neutral']['count'], 
                                       sentiment_dist['negative']['count']],
                                marker=dict(colors=['#4CAF50', '#FFC107', '#F44336'])
                            )])
                            fig.update_layout(title="توزيع المشاعر", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    st.markdown("---")
                    query_terms = sys.preprocess(query)
                    for idx, (doc_id, score) in enumerate(results, 1):
                        sentiment_emoji = {0: "😞", 1: "😐", 2: "😊"}
                        sentiment_label = {0: "سلبي", 1: "محايد", 2: "إيجابي"}
                        doc_sentiment = sys.sentiments.get(doc_id, 1)
                        with st.expander(f"نتيجة {idx} | {sentiment_emoji[doc_sentiment]} {sentiment_label[doc_sentiment]} | الدرجة: {score:.4f}"):
                            highlighted = sys.highlight_terms(sys.doc_texts[doc_id], query_terms)
                            st.markdown(highlighted, unsafe_allow_html=True)
                            st.caption(f"رقم الوثيقة: {doc_id}")
                else:
                    st.warning("لم يتم العثور على نتائج")
        else:
            st.warning("الرجاء إدخال كلمات للبحث")

if __name__ == "__main__":
    main()