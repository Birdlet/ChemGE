import nltk
import numpy as np
import utilities as util
import scorer
from config import GCFG


class optimizer():
    def __init__(self):
        self._population = []
        self._grammer = "zinc_grammer"
        self._GCFG = GCFG
        self._tokenize = self._get_zinc_tokenizer(self._GCFG)
        self._parser = nltk.ChartParser(self._GCFG)

    def _get_zinc_tokenizer(self, cfg):
        long_tokens = [a for a in cfg._lexical_index.keys() if len(a) > 1]
        replacements = ['$', '%', '^']
        assert len(long_tokens) == len(replacements)
        for token in replacements:
            assert token not in cfg._lexical_index

        def tokenize(smiles):
            for i, token in enumerate(long_tokens):
                smiles = smiles.replace(token, replacements[i])
            tokens = []
            for token in smiles:
                try:
                    ix = replacements.index(token)
                    tokens.append(long_tokens[ix])
                except Exception:
                    tokens.append(token)
            return tokens
        return tokenize


    def encode(self, smiles):
        tokens = self._tokenize(smiles)
        parser = nltk.ChartParser(self._GCFG)
        # if you use Python 2: parse_tree = parser.parse(tokens).next()
        parse_tree = parser.parse(tokens).__next__()
        productions_seq = parse_tree.productions()
        productions = GCFG.productions()
        prod_map = {}
        for ix, prod in enumerate(productions):
            prod_map[prod] = ix
        indices = np.array([prod_map[prod] for prod in productions_seq], dtype=int)
        return indices

    def decode(self, rule):
        productions = self._GCFG.productions()
        prod_seq = [productions[i] for i in rule]

        # prod_to_eq
        seq = [prod_seq[0].lhs()]
        for prod in prod_seq:
            if str(prod.lhs()) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod.lhs():
                    seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                    break
        try:
            smiles = ''.join(seq)
        except Exception:
            smiles = ''
        return smiles


    def CFGtoGene(self, prod_rules, max_len=-1):
        gene = []
        for r in prod_rules:
            lhs = GCFG.productions()[r].lhs()
            possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                            if rule.lhs() == lhs]
            gene.append(possible_rules.index(r))
        if max_len > 0:
            if len(gene) > max_len:
                gene = gene[:max_len]
            else:
                gene = gene + [np.random.randint(0, 256)
                            for _ in range(max_len-len(gene))]
        return gene


    def GenetoCFG(self, gene):
        prod_rules = []
        stack = [GCFG.productions()[0].lhs()]
        for g in gene:
            try:
                lhs = stack.pop()
            except Exception:
                break
            possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                            if rule.lhs() == lhs]
            rule = possible_rules[g % len(possible_rules)]
            prod_rules.append(rule)
            rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal)
                        and (str(a) != 'None'),
                        self._GCFG.productions()[rule].rhs())
            stack.extend(list(rhs)[::-1])
        return prod_rules
    
    
    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)


    def optimize(self):
        pass


class optimizerJ(optimizer):
    def __init__(self):
        optimizer.__init__(self)


    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)


    def optimize(self, smiles, log = None, mu=32, lam=64, generation=1000, seed=0, verbose=True):
        np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(s) for s in initial_smiles]
        initial_genes = [self.CFGtoGene(self.encode(s), max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = [scorer.score(s) for s in initial_smiles]

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)

                c_smiles = util.canonicalize(self.decode(self.GenetoCFG(c_gene)))
                if c_smiles not in all_smiles:
                    c_score = scorer.score(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size =len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result


class optimizerRDock(optimizer):
    def __init__(self):
        optimizer.__init__()

    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)



    def optimize(self, smiles, log = None, mu=32, lam=64, generation=1000, seed=0, verbose=True):
        np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(s) for s in initial_smiles]
        initial_genes = [self.CFGtoGene(self.encode(s), max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = [scorer.score(s) for s in initial_smiles]

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)

                c_smiles = util.canonicalize(self.decode(self.GenetoCFG(c_gene)))
                if c_smiles not in all_smiles:
                    c_score = scorer.score(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size =len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result



class optimizerVina(optimizer):
    def __init__(self):
        optimizer.__init__(self)

    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)



    def optimize(self, smiles, vina_conf, vina_log_path, \
        log = None, mu=32, lam=64, generation=1000, seed=0, verbose=True):

        np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(s) for s in initial_smiles]
        initial_genes = [self.CFGtoGene(self.encode(s), max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = []
        print(r"|0%--------------------50%-------------------100%|")
        for i, s in enumerate(initial_smiles):
            scorer.score_vina(s, conf_path=vina_conf, log_path=vina_log_path)
            print("*"*int(50*i/(mu+lam)), end='\r')
        print()

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]
    

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)

                c_smiles = util.canonicalize(self.decode(self.GenetoCFG(c_gene)))
                if c_smiles not in all_smiles:
                    c_score = scorer.score(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size =len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result
