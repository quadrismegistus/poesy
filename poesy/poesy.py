## encoding=utf-8
from __future__ import division
import sys,os,codecs
import cPickle,random,numpy as np
import logging
from collections import Counter
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
PATH_RHYME_SCHEMES = os.path.join(os.path.dirname(__file__),'schemes','rhyme_schemes.txt')
RHYME_SCHEMES=None
METER='meter_default'

MAX_RHYME_DIST=5

def test():
	# poemtxt="""FROM fairest creatures we desire increase,
	# That thereby beauty's rose might never die,
	# But as the riper should by time decease,
	# His tender heir might bear his memory:
	# But thou, contracted to thine own bright eyes,
	# Feed'st thy light'st flame with self-substantial fuel,
	# Making a famine where abundance lies,
	# Thyself thy foe, to thy sweet self too cruel.
	# Thou that art now the world's fresh ornament
	# And only herald to the gaudy spring,
	# Within thine own bud buriest thy content
	# And, tender churl, makest waste in niggarding.
	# Pity the world, or else this glutton be,
	# To eat the world's due, by the grave and thee."""
	poemtxt="""Since brass, nor stone, nor earth, nor boundless sea,
	But sad mortality o'er-sways their power,
	How with this rage shall beauty hold a plea,
	Whose action is no stronger than a flower?
	O, how shall summer's honey breath hold out
	Against the wreckful siege of battering days,
	When rocks impregnable are not so stout,
	Nor gates of steel so strong, but Time decays?
	O fearful meditation! where, alack,
	Shall Time's best jewel from Time's chest lie hid?
	Or what strong hand can hold his swift foot back?
	Or who his spoil of beauty can forbid?
	O, none, unless this miracle have might,
	That in black ink my love may still shine bright."""

	p=Poem(poemtxt)
	return p.rhymed




class Poem(object):
	def __init__(self,txt=None,id=None,title=None,fn=None,fn_encoding='utf-8',meter=METER):
		if fn and not txt:
			if os.path.exists(fn):
				with codecs.open(fn,encoding=fn_encoding) as f:
					txt=f.read()

		if not txt: raise ValueError("Neither a txt string object was passed nor a working filename through fn=")
		self.id=hash(txt) if not id else id
		self.meter=meter


		txt=txt.strip()
		txt=txt.replace('\r\n','\n').replace('\r','\n')
		while '\n\n\n' in txt: txt=txt.replace('\n\n\n','\n\n')

		##
		# Unicode tricks
		txt=txt.replace(u'è','-e')
		##

		self.title=title if title else txt.split('\n')[0].strip()
		self.txt=txt

		lined={}
		linenum=0
		stanza_lens=[]
		for stanza_i,stanza in enumerate(txt.split('\n\n')):
			stanza=stanza.strip()
			num_line_in_stanza=0

			for line_i,line in enumerate(stanza.split('\n')):
				line=line.strip()
				if not line: continue
				num_line_in_stanza+=1
				linenum+=1
				stanzanum=stanza_i+1
				lineid=(linenum, stanzanum)
				linetext=line
				linetext=linetext.replace('& ','and ')
				linetext=linetext.replace(u'—',' ')
				linetext=linetext.replace(u'&ebar;','e')
				lined[lineid]=linetext

			stanza_lens+=[num_line_in_stanza]

		self._stanza_length=stanza_lens[0] if len(set(stanza_lens))==1 else None

		self.lined=lined
		self.numLines=len(self.lined)
		self.genn=True

	@property
	def lines(self):
		return [v for k,v in sorted(self.lined.items())]


	@property
	def stanzas(self):
		s2i={}
		for li in sorted(self.lined.keys()):
			s=tuple(li[1:])
			if not s in s2i: s2i[s]=[]
			s2i[s]+=[li]
		return [l for s,l in sorted(s2i.items())]

	@property
	def stanzas_prosodic(self):
		s2i={}
		for li in sorted(self.prosodic.keys()):
			if not self.prosodic[li].words(): continue
			s=tuple(li[1:])
			if not s in s2i: s2i[s]=[]
			s2i[s]+=[li]
		return [l for s,l in sorted(s2i.items())]



	@property
	def stanza_length(self):
		"""
		Returns invariable stanza length as an integer.
		If variable stanza lengths, returns None.
		"""
		if hasattr(self,'_stanza_length'): return self._stanza_length
		stanza_lens = [len(st) for st in self.stanzas]
		lenfreq=Counter(stanza_lens)
		mostcommonlen=[_k for _k in sorted(lenfreq,key=lambda __k: -lenfreq[__k])][0]

		if not stanza_lens or len(stanza_lens)==1:
			self._stanza_length=None # return None if variable stanza lengths
		else:
			self._stanza_length=mostcommonlen
		return self._stanza_length

	@property
	def firstline(self):
		return self.lines[0]


	## METER

	@property
	def hood_dist(self):
		## HOOD DIST
		x=.175
		y=.5

		def distance(p0, p1):
			import math
			return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

		try:
			y2=float(self.meterd['meter_perc_lines_fourthpos_s'])
			x2=float(self.meterd['meter_mpos_ww'])
		except (ValueError,KeyError) as e:
			return None
		dist=distance((x,y),(x2,y2))
		return dist

	@property
	def total_viols(self):
		return self.statd['meter_constraint_TOTAL']

	@property
	def meterd(self):
		"""
		Return dictionary with metrical annotations.
		"""
		datad={}
		all_mstrs=[]
		ws_ends=[]
		ws_starts=[]
		ws_fourths=[]

		## COLLECT STATS ON metrically weak/strong in various positions
		viold={}
		for i,line in sorted(self.prosodic.items()):
			bp=line.bestParses()
			if not bp: continue
			mstrs=[]
			pure_parse=''
			for parse in bp:
				for mpos in parse.positions:
					for ck,cv in mpos.constraintScores.items():
						ck=ck.name
						if not ck in viold: viold[ck]=[]
						viold[ck]+=[0 if not cv else 1]
					mstr=''.join([mpos.meterVal for n in range(len(mpos.slots))])
					pure_parse+=mstr
					mstrs+=[mstr]
			line_mstr='|'.join(mstrs)
			line_parse="||".join(unicode(p) for p in bp)
			ws_starts += [pure_parse[0]]
			ws_ends += [pure_parse[-1]]
			ws_fourths += [pure_parse[3:4]]
			all_mstrs+=mstrs


		mstr_freqs=toks2freq(all_mstrs,tfy=True)
		for k,v in mstr_freqs.items(): datad['mpos_'+k]=v
		for k,v in toks2freq(ws_ends,tfy=True).items(): datad['perc_lines_ending_'+k]=v
		for k,v in toks2freq(ws_starts,tfy=True).items(): datad['perc_lines_starting_'+k]=v
		for k,v in toks2freq(ws_fourths,tfy=True).items(): datad['perc_lines_fourthpos_'+k]=v


		## DECIDE WHETHER TERNARY / BINARY FOOT
		d=datad
		d['type_foot']='ternary' if d.get('mpos_ww',0)>0.175 else 'binary'

		## DECIDE WHETHER INITIAL / FINAL-HEADED
		if d['type_foot']=='ternary':
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)>0.5 else 'final'
		else:
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)<0.5 else 'final'

		## PUT 2 TOGETHER TO DECIDE anapestic / dactylic / trochaic / iambic
		x=(d['type_foot'],d['type_head'])
		if x==('ternary','final'):
			d['type_scheme']='anapestic'
		elif x==('ternary','initial'):
			d['type_scheme']='dactylic'
		elif x==('binary','initial'):
			d['type_scheme']='trochaic'
		else:
			d['type_scheme']='iambic'


		### METRICAL AMBIGUITY
		ambig = []
		avg_linelength=[]
		avg_parselength=[]
		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			line_numparses=[]
			line_parselen=0
			if not ap: continue
			for parselist in ap:
				numparses=len(parselist)
				parselen=len(parselist[0].str_meter())
				avg_parselength+=[parselen]
				line_parselen+=parselen
				line_numparses+=[numparses]

			import operator
			avg_linelength+=[line_parselen]
			ambigx=reduce(operator.mul, line_numparses, 1)
			ambig+=[ambigx]
		d['ambiguity']=np.mean(ambig) if ambig else ''
		d['length_avg_line']=np.mean(avg_linelength) if avg_linelength else ''
		d['length_avg_parse']=np.mean(avg_parselength) if avg_parselength else ''


		## TOTAL METRICAL VIOLATIONS
		sumviol=0
		for ck,cv in viold.items():
			avg=np.mean(cv) if cv else ''
			d['constraint_'+ck.replace('.','_')]=avg
			sumviol+=avg

		d['constraint_TOTAL']=sumviol
		return datad


	def __str__(self):
		"""
		Return a string version of poem: its ID
		"""
		return self.id

	def limit(self,N,preserve_stanza=True):
		""" Limit number of lines to first N """
		newd={}
		if self.stanza_length:
			lineids=[]
			for lineids_in_stanza in self.stanzas:
				lineids+=lineids_in_stanza
				if len(lineids)>=N:
					break
			for li in lineids:
				newd[li]=self.lined[li]
		else:
			for i,(k,v) in enumerate(sorted(self.lined.items())):
				if i>=N: break
				newd[k]=v
		self.lined=newd

	@property
	def indices(self):
		return sorted(self.lined.keys())


	@property
	def meterd(self):
		datad={}
		all_mstrs=[]
		ws_ends=[]
		ws_starts=[]
		ws_fourths=[]

		viold={}
		for i,line in sorted(self.prosodic.items()):
			bp=line.bestParses()
			if not bp: continue
			mstrs=[]
			pure_parse=''
			for parse in bp:
				for mpos in parse.positions:
					for ck,cv in mpos.constraintScores.items():
						ck=ck.name
						if not ck in viold: viold[ck]=[]
						viold[ck]+=[0 if not cv else 1]

					mstr=''.join([mpos.meterVal for n in range(len(mpos.slots))])
					pure_parse+=mstr
					mstrs+=[mstr]
			line_mstr='|'.join(mstrs)
			line_parse="||".join(unicode(p) for p in bp)
			ws_starts += [pure_parse[0]]
			ws_ends += [pure_parse[-1]]
			ws_fourths += [pure_parse[3:4]]

			all_mstrs+=mstrs

		mstr_freqs=toks2freq(all_mstrs,tfy=True)
		for k,v in mstr_freqs.items(): datad['mpos_'+k]=v

		for k,v in toks2freq(ws_ends,tfy=True).items(): datad['perc_lines_ending_'+k]=v
		for k,v in toks2freq(ws_starts,tfy=True).items(): datad['perc_lines_starting_'+k]=v
		for k,v in toks2freq(ws_fourths,tfy=True).items(): datad['perc_lines_fourthpos_'+k]=v

		d=datad
		#d['type_foot']='ternary' if d.get('mpos_ww',0)>0.15 and d.get('mpos_w',0)<.35 else 'binary'
		d['type_foot']='ternary' if d.get('mpos_ww',0)>0.175 else 'binary'

		if d['type_foot']=='ternary':
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)>0.5 else 'final'
		else:
			d['type_head']='initial' if d.get('perc_lines_fourthpos_s',0)<0.5 else 'final'


		x=(d['type_foot'],d['type_head'])
		if x==('ternary','final'):
			d['type_scheme']='anapestic'
		elif x==('ternary','initial'):
			d['type_scheme']='dactylic'
		elif x==('binary','initial'):
			d['type_scheme']='trochaic'
		else:
			d['type_scheme']='iambic'


		### AMBIGUITY
		ambig = []
		avg_linelength=[]
		avg_parselength=[]
		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			line_numparses=[]
			line_parselen=0
			if not ap: continue
			for parselist in ap:
				numparses=len(parselist)
				parselen=len(parselist[0].str_meter())
				avg_parselength+=[parselen]
				line_parselen+=parselen
				line_numparses+=[numparses]

			import operator
			avg_linelength+=[line_parselen]
			ambigx=reduce(operator.mul, line_numparses, 1)
			ambig+=[ambigx]
		d['ambiguity']=np.mean(ambig) if ambig else ''
		d['length_avg_line']=np.mean(avg_linelength) if ambig else ''
		d['length_avg_parse']=np.mean(avg_parselength) if ambig else ''


		## VIOLATIONS
		sumviol=0
		for ck,cv in viold.items():
			avg=np.mean(cv) if cv else ''
			d['constraint_'+ck.replace('.','_')]=avg
			sumviol+=avg

		d['constraint_TOTAL']=sumviol
		return datad

	#@property
	#def meter(self):
	#	return self.meterd['type_scheme']

	@property
	def parsed(self):
		if hasattr(self,'_parsed'): return self._parsed
		meterd=self.meterd

		"""
		Tie-breaker logic:
		anapestic --> maximize ww, start with w
		trochaic --> minimize ww, start with s
		dactylic --> maximize ww, start with s
		iambic --> minimize ww, start with w
		"""
		self._parsed=parsed={}
		def sort_ties(ties, meterd):
			ww_factor=-1 if meterd['type_foot']=='ternary' else 1
			wstart_factor=1 if meterd['type_head']=='initial' else -1
			def _sort_tuple(P):
				num_ww=sum([int(mpos.mstr=='ww') for mpos in P.positions])
				zero_means_starts_with_w=int(P.positions[0].mstr[0]!='w')
				return (num_ww,zero_means_starts_with_w)

			ties.sort(key=lambda P: (ww_factor*_sort_tuple(P)[0], wstart_factor*_sort_tuple(P)[1]))

		for i,line in sorted(self.prosodic.items()):
			ap=line.allParses()
			if not ap: continue
			parsed[i]=lineparses=[]
			for parselist in ap:	# first level of list is punctuation breaks
				## here is where we decide among parses based on which maximizes metrical scheme
				## only if there are ties
				parselist.sort(key=lambda P: P.totalScore)
				lowestScore=parselist[0].totalScore
				ties=[P for P in parselist if P.totalScore==lowestScore]
				if len(ties)>1: sort_ties(ties,meterd)
				lineparses+=[ties[0]]

		return parsed

	def parse(self,lim=None,meter=None):
		if not meter: meter=self.meter
		if hasattr(self,'_parsed') and self._parsed:
			return
		for _i,(li,line) in enumerate(self.prosodic.items()):
			if lim and _i>=lim: break
			line.parse(meter=meter)
		self._parsed=True

	def get_schemed(self,beat=True):
		scheme,sdiff=self.get_scheme(beat=beat,return_diff=True)
		dx={}
		dx['scheme']=scheme
		dx['scheme_type']=self.schemetype(scheme)
		dx['scheme_repr']=self.scheme_repr(dx['scheme_type'], dx['scheme'],beat=beat)
		dx['scheme_length']=len(scheme)
		dx['scheme_diff']=sdiff
		return dx

	@property
	def schemed(self):
		return self.get_schemed(beat=True)

	@property
	def schemed_syll(self):
		return self.get_schemed(beat=False)

	@property
	def schemed_beat(self):
		return self.get_schemed(beat=True)

	@property
	def lineld(self):
		old=[]
		self.parse()
		self.rhyme_net()

		for lineid in sorted(self.lined):
			odx={
				'lineid':lineid,
				'#line':lineid[0],
				'#stanza':lineid[1],
				'#line_in_stanza':self.linenums_bystanza[lineid],
				'lineid2':'%s.%s' % (lineid[1],self.linenums_bystanza[lineid]),

				'line':self.lined[lineid],
				'parse':self.prosodic[lineid].parse_str(viols=True),
				'num_parses':self.numparses[lineid],

				'num_sylls':self.linelengths[lineid],
				'num_feet':self.linelengths_bybeat[lineid],

				'rhyme':self.rhymes[lineid],
			}
			old+=[odx]
		return old

	def show(self):
		"""Show annotations
		"""
		ostr=[]
		self.parse()
		self.rhyme_net()
		stanzanow=None

		lineid2linestr=dict((lineid,line.parse_str(viols=False)) for lineid,line in sorted(self.prosodic.items()))
		maxlinelen=max(len(l) for l in lineid2linestr.values())
		linelen=maxlinelen+5

		for lineid,line in sorted(self.prosodic.items()):
			rimestr=self.rhymes[lineid]
			linestr=line.parse_str(viols=False)
			linenum=self.linenums_bystanza[lineid]
			stanzanum=lineid[1]
			beatlen=self.linelengths_bybeat[lineid]
			sylllen=self.linelengths[lineid]
			if stanzanow!=stanzanum:
				if ostr: ostr+=['']
				stanzanow=stanzanum

			oline='({stanza}.{linenum}) {line:<{linelen}} [{rime}] [{beat}/{syll}]'.format(
				linelen=linelen,
				rime=rimestr,
				line=linestr,
				stanza=stanzanum,
				linenum=linenum,
				beat=beatlen,
				syll=sylllen)
			ostr+=[oline]
		return '\n'.join(ostr)

	def summary(self,header=['lineid2', 'parse', 'rhyme', 'num_feet', 'num_sylls','num_parses']):
		colnames={
		'num_feet':'#feet',
		'num_sylls':'#syll',
		'lineid':'(#ln,#st)',
		'lineid2':'(#s,#l)',
		'rhyme':'rhyme',
		'parse':'parse',
		'num_parses':'#parse'
		}

		from tabulate import tabulate
		data=[]
		stanzanow=None
		for row in self.lineld:
			if stanzanow is None: stanzanow=row['#stanza']
			if stanzanow!=row['#stanza']:
				data+=['']
				stanzanow=row['#stanza']

			datarow=[row[h] for h in header]
			data+=[datarow]
		cols=[colnames.get(h,h) for h in header]
		table=tabulate(data,headers=cols)

		schemestr1='meter: {meter}\nfeet: {feet}\nsyllables: {syll}\nrhyme: {rhymename} {rhymescheme}'.format(
			meter=self.statd['meter_type_scheme'].title(),
			feet=self.statd['beat_scheme_repr'],
			syll=self.statd['syll_scheme_repr'],
			rhymename=self.statd['rhyme_scheme_name'],
			rhymescheme='(%s)' % self.statd['rhyme_scheme_form'] if self.statd['rhyme_scheme_form'] else '',
		)

		ostr=table+'\n\n\nestimated schema\n----------\n'+schemestr1
		print ostr

	@property
	def statd(self):
		if not hasattr(self,'_statd') or not self._statd:
			dx=self._statd={}

			## Scheme
			for x,y in [('beat',True), ('syll',False)]:
				sd=self.get_schemed(beat=y)
				for sk,sv in sd.iteritems():
					dx[x+'_'+sk]=sv

			## Length
			dx['num_lines']=self.numLines

			## Meter
			for k,v in self.meterd.items(): dx['meter_'+k]=v

			## Rhyme
			for k,v in self.rhymed.items():
				if k=='rhyme_schemes' and v: v=v[-5:]
				dx[k]=v
		return self._statd

	def schemetype(self,scheme):
		if len(scheme)==1: return 'Invariable'
		if len(scheme)==2: return 'Alternating'
		return 'Complex'

	def scheme_repr(self,schemetype,scheme,beat=False):
		#if schemetype=='Complex': return 'Complex'
		#prefix='Alt_' if schemetype=='Alternating' else 'Inv_'
		if beat and schemetype!='Complex': scheme=[BEATNAMES.get(sx,sx) for sx in scheme]
		if schemetype=='Invariable':
			return scheme[0]
		# elif schemetype=='Complex':
		#
		# 	return
		schemedetails='('+'-'.join(unicode(sx) for sx in scheme)+')'
		return schemetype+' '+schemedetails.lower()


	def get_scheme(self,beat=True,return_diff=False,encourage_invariable=True):
		stanza_length=self.stanza_length
		if beat:
			lengths=[v for k,v in sorted(self.linelengths_bybeat.items())]
		else:
			lengths=[v for k,v in sorted(self.linelengths.items())]

		num_lines=len(lengths)
		min_length,max_length=min(lengths),max(lengths)
		abs_diff_in_lengths=abs(min_length-max_length)

		if beat:
			isVariable=True if abs_diff_in_lengths>2 else False
		else:
			isVariable=True if abs_diff_in_lengths>4 else False


		min_seq_length=1 # if not isVariable else 2
		try_lim=10
		max_seq_length=stanza_length if stanza_length else 12


		def measure_diff(l1,l2,beat=False):
			min_l=min([len(l1),len(l2)])
			l1=l1[:min_l]
			l2=l2[:min_l]
			"""
			print len(l1),len(l2)
			print '  '.join(unicode(x) for x in l1)
			print '  '.join(unicode(x) for x in l2)
			print '  '.join(unicode(abs(x1-x2)) for x1,x2 in zip(l1,l2))
			#"""
			diff=0
			for x1,x2 in zip(l1,l2):
				diff+=abs(x1-x2)
			return diff if not beat else diff*2

		combo2diff={}
		best_combo=None
		best_diff=None

		best_combos=[]
		best_lim=100
		poem_length=self.numLines
		for seq_length in range(min_seq_length,int(max_seq_length)+1):
			if seq_length>poem_length: break
			if stanza_length and stanza_length % seq_length: continue
			if poem_length and poem_length % seq_length: continue
			if seq_length>try_lim: break
			num_reps=num_lines/seq_length

			average_length_per_pos=dict((s_i,[]) for s_i in range(seq_length))
			for l_i,l_x in enumerate(lengths):
				average_length_per_pos[l_i % seq_length] += [l_x]
			for k,v in average_length_per_pos.items():
				median=np.median(v) if len(v)>1 else v[0]
				average_length_per_pos[k]=int(median)

			SOME_possibilities = [[rx for rx in range(average_length_per_pos[x_i]-1,average_length_per_pos[x_i]+2)] for x_i,x in enumerate(range(seq_length))]
			combo_possibilities = list(product(*SOME_possibilities))
			for combo in combo_possibilities:
				if len(combo)>1 and len(set(combo))==1: continue
				model_lengths=[]
				while len(model_lengths)<=len(lengths):
					for cx in combo: model_lengths+=[cx]
				model_lengths=model_lengths[:len(lengths)]

				diff_in_lengths=abs(len(lengths) - len(model_lengths))
				diff=measure_diff(lengths, model_lengths, beat=beat)
				if not beat:
					diff+=sum([5 if seq_x%2 else 0 for seq_x in combo])
				if encourage_invariable:
					diff+=1 if len(set(combo))>1 else 0

				diff=diff


				if len(best_combos)<best_lim or diff<max([_d for _c,_d in best_combos]):
					best_combos+=[(combo,diff)]
					best_combos=sorted(best_combos,key=lambda _lt: _lt[1])[:best_lim]

				if best_diff==None or diff<best_diff:
					best_diff=diff
					best_combo=combo

				elif best_combo and diff<=best_diff:
					if len(combo)<len(best_combo):
						best_combo=combo
						best_diff=diff
					elif np.mean(combo) if combo else 0 > np.mean(best_combo) if best_combo else 0:
						best_diff=diff
						best_combo=combo


		self._scheme=best_combo
		self._scheme_diff=best_diff
		if return_diff:
			return best_combo,best_diff

		return best_combo

	@property
	def scheme(self):
		return self.get_scheme(beat=True)

	@property
	def prosodic(self):
		if not hasattr(self,'_prosodic'):
			import prosodic as p
			p.config['print_to_screen']=0
			self._prosodic=pd={}
			numlines=len(self.lined)
			for _i,(i,line) in enumerate(sorted(self.lined.items())):
				line=line.replace('-',' ').replace("'","").strip()
				pd[i]=p.Text(line,meter=self.meter)
		return self._prosodic

	@property
	def linelengths(self):
		if not hasattr(self,'_linelengths'):
			self._linelengths=dx={}
			for lineid,line in sorted(self.prosodic.items()):
				dx[lineid]=len(line.syllables())
		return self._linelengths

	@property
	def linelengths_bybeat(self):
		if not hasattr(self,'_linelengths_bybeat'):
			self.parse()
			self._linelengths_bybeat=dx={}
			for lineid,line in sorted(self.prosodic.items()):
				dx[lineid]=num_beats(line)
		return self._linelengths_bybeat

	@property
	def linelength(self):	# median line length
		if not hasattr(self,'_linelength'):
			self._linelength=np.median(self.linelengths.values())
		return self._linelength


	## Meter

	@property
	def numparses(self):
		if not hasattr(self,'_numparses'):
			self._numparses=npd={}
			for lineid in self.prosodic:
				line=self.prosodic[lineid]
				npd[lineid]=line.ambiguity #@IMPORTANT look at this code in prosodic/lib/Text
		return self._numparses





	@property
	def linenums(self):
		return dict((lineid,lineid[0]) for lineid in self.lined)

	@property
	def stanzanums(self):
		return dict((lineid,lineid[1]) for lineid in self.lined)


	@property
	def linenums_bystanza(self):
		"""
		Within stanza numberings
		"""
		if not hasattr(self,'_linenums'):
			rd=self._linenums={}
			stanzanow=None
			linenum=0
			for lineid,line in sorted(self.lined.items()):
				if lineid[1]!=stanzanow:
					stanzanow=lineid[1]
					linenum=0
				linenum+=1
				rd[lineid]=linenum
		return self._linenums

	## RHYME
	@property
	def rhymes(self):
		if not hasattr(self,'_rhymes'):
			rd=self._rhymes={}
			for lineid,line in sorted(self.lined.items()):
				try:
					rimestr=self.rhyme_ids[lineid[0]-1]
				except IndexError:
					rimestr='?'
				rd[lineid]=rimestr
		return self._rhymes

	@property
	def rhyme_ids(self):
		if not hasattr(self,'_rhyme_ids'):
			self.rhyme_net()
			self._rhyme_ids=nums2scheme(self.rime_ids)
		return self._rhyme_ids

	@property
	def rhymed(self):
		if hasattr(self,'_rhymed'): return self._rhymed
		self._rhymed=odx={}
		odx['rhyme_scheme']=''
		odx['rhyme_scheme_name']=''
		odx['rhyme_scheme_form']=''
		odx['rhyme_scheme_accuracy']=''
		self.rhyme_net()
		for k,v in self.discover_rhyme_scheme(self.rime_ids).items(): odx[k]=v
		return odx


	def rhyme_net(self,toprint=False,force=False):
		if not force and hasattr(self,'rhymeG') and self.rhymeG: return self.rhymeG
		W=4
		import networkx as nx
		G=nx.DiGraph()
		tried=set()
		old=[]
		for stnum,stanza in enumerate(self.stanzas_prosodic):
			for i,lineid1 in enumerate(stanza):
				prev_lines=stanza[i-W if i-W > 0 else 0:i]
				next_lines=stanza[i+1:i+1+W]
				for lineid2 in prev_lines + next_lines:
					line1=self.prosodic[lineid1]
					line2=self.prosodic[lineid2]
					node1=unicode(lineid1[0]).zfill(6)+': '+self.lined[lineid1]
					node2=unicode(lineid2[0]).zfill(6)+': '+self.lined[lineid2]
					dist=line1.rime_distance(line2)

					odx={'node1':node1,'node2':node2, 'dist':dist, 'lineid1':lineid1, 'lineid2':lineid2}
					old+=[odx]
					G.add_edge(node1,node2,weight=dist)

		## ASSIGN RIME IDS
		self.rime_ids=ris=[]
		node2num={}
		nnum=1
		overlaps=set()
		weights=[]
		toprintstr=[]
		for node in sorted(G.nodes()):
			#print "NODE",node
			edged=G.edge if hasattr(G,'edge') else G.adj  # diff versions of networkx?
			neighbors=sorted(edged[node].keys(),key=lambda n2: G[node][n2]['weight'])
			#neighbors=[n for n in neighbors if n>node]
			closest_neighbor=neighbors[0]
			#print 'closest neighbor:',closest_neighbor
			#print
			closest_weight=G[node][closest_neighbor]['weight']
			weights+=[closest_weight]
			if closest_weight > MAX_RHYME_DIST:
				#nodenum=nnum
				nodenum=0
				node2num[node]=nodenum
				nnum+=1
			if node in node2num:
				nodenum=node2num[node]
			elif closest_neighbor in node2num:
				nodenum=node2num[closest_neighbor]
				node2num[node]=nodenum
				#if n in node2num:
				#	nodenum=node2num[n]
				#	break
			else:
				node2num[node]=nnum
				node2num[closest_neighbor]=nnum
				nodenum=nnum
				nnum+=1

			if toprint: toprintstr+=[node+'\t'+unicode(nodenum)]
			G.node[node]['rime_id']=nodenum
			ris+=[nodenum]

		self.rhyme_scheme=''
		self.rhyme_scheme_accuracy=''
		self.rhyme_weight_avg=np.mean(weights) if weights else ''

		self.rhymeG=G
		if toprint:
			print
			print '\n'.join(toprintstr)
			print

		# @NEW
		#self.rime_ids=transpose(self.rime_ids)
		return G


	def discover_rhyme_scheme(self,rime_ids):
		global RHYME_SCHEMES
		if not RHYME_SCHEMES:
			RHYME_SCHEMES=[d for d in read_tsv(PATH_RHYME_SCHEMES)]

		odx={'rhyme_scheme':None, 'rhyme_scheme_accuracy':None, 'rhyme_schemes':None}
		if not rime_ids: return odx

		def translate_slice(slice):
			unique_numbers=set(slice)
			unique_numbers_ordered=sorted(list(unique_numbers))
			for i,number in enumerate(slice):
				if number==0: continue
				slice[i] = unique_numbers_ordered.index(number) + 1
			return slice

		def scheme2edges(scheme):
			id2pos={}
			for i,x in enumerate(scheme):
				# x is a rhyme id, i is the position in the scheme
				if x==0: continue
				if not x in id2pos: id2pos[x]=[]
				id2pos[x]+=[i]

			rhymes=[]
			for x in id2pos:
				if len(id2pos[x])>1:
					for a,b in product(id2pos[x], id2pos[x]):
						if a>=b: continue
						rhymes+=[(a,b)]
			return rhymes

		def test_edges(scheme_exp,scheme_obs):
			edges_exp=scheme2edges(scheme_exp)
			edges_obs=scheme2edges(scheme_obs)
			set_edges_exp = set(edges_exp)
			set_edges_obs = set(edges_obs)

			logging.debug(('Expecting these edges:', edges_exp))
			logging.debug(('Found these edges:', edges_obs))
			logging.debug(('These edges unexpectedly present:',sorted(list(set_edges_obs-set_edges_exp))))
			logging.debug(('These edges unexpectedly absent:',sorted(list(set_edges_exp-set_edges_obs))))

			# @NEW
			#return np.mean([int(x!=y) for x,y in zip(scheme_exp,scheme_obs)])
			divisor=float(len(set_edges_exp | set_edges_obs))
			jaccard = len(set_edges_exp & set_edges_obs) / divisor if divisor else 0
			return jaccard

		def test_scheme(scheme):
			logging.debug(("scheme:",scheme))
			scheme_nums=scheme2nums(scheme)
			slices=slicex(rime_ids,slice_length=len(scheme_nums),runts=True)
			matches=[]

			logging.debug((">> RIME IDS:",rime_ids))
			did_not_divide=0
			for si,slice in enumerate(slices):

				tslice=translate_slice(slice)
				#match = int(tslice == scheme_nums)
				totest=scheme_nums[:len(tslice)]
				match = test_edges(totest, tslice)
				#match = test_edges(scheme_nums, tslice)

				if len(scheme_nums) != len(tslice):
					did_not_divide+=1
				logging.debug((">>",si,"slice, looking for:",scheme_nums,"and found:",tslice))
				logging.debug((">> MATCH:", match))
				logging.debug('')
				matches+=[match]
				# translate down

			#print matches
			match_score=(np.mean(matches) if matches else 0) - did_not_divide
			return match_score

		scheme_scores={}



		for schemed in RHYME_SCHEMES:
			#if not 'shakespeare' in schemed['form']: continue
			logging.debug(('>> TESTING SCHEME:',schemed['Form']))
			scheme=schemed['Scheme']
			scheme_score=test_scheme(scheme)
			#if scheme_score:
			scheme_scores[(schemed['Form'],scheme)]=scheme_score

		odx['rhyme_schemes']=sorted(scheme_scores.items(),key=lambda lt: -lt[1])[:5]
		#for scheme,scheme_score in sorted(scheme_scores.items(),key=lambda lt: (lt[1],-len(lt[0]))):
		# @NEW jaccard
		for scheme,scheme_score in sorted(scheme_scores.items(),key=lambda lt: (-lt[1],-len(lt[0]))):
			odx['rhyme_scheme']=scheme
			odx['rhyme_scheme_name']=scheme[0]
			odx['rhyme_scheme_form']=scheme[1]
			odx['rhyme_scheme_accuracy']=scheme_score
			break

		if not odx['rhyme_scheme']: odx['rhyme_scheme']='Unknown'
		return odx

	@property
	def isSonnet(self):
		# @TODO
		if len(self.lined)!=14: return False
		if not int(self.linelength) in [9,10,11]: return False
		if not 'sonnet' in self.rhymed['rhyme_scheme'][0].lower(): return False
		return True

	@property
	def isShakespeareanSonnet(self):
		# @TODO
		if len(self.lined)!=14: return False
		if not int(self.linelength) in [9,10,11]: return False
		if not 'sonnet' in self.rhymed['rhyme_scheme'][0].lower(): return False
		if not 'shakespeare' in self.rhymed['rhyme_scheme'][0].lower(): return False
		#if self.rime_ids != [2, 1, 2, 1, 3, 4, 3, 4, 5, 6, 5, 6, 7, 7]: return False
		return True


def num_beats(line):
	return len([mpos for mpos in line.bestParses()[0].positions if mpos.meterVal=='s'])



def transpose(slice):
	unique_numbers=set(slice)
	unique_numbers_ordered=sorted(list(unique_numbers))
	for i,number in enumerate(slice):
		if number==0: continue
		slice[i] = unique_numbers_ordered.index(number) + 1
	return slice

def scheme2nums(scheme):
	scheme=scheme.replace(' ','')
	scheme_length=len(scheme)
	alphabet='abcdefghijklmnopqrstuvwxyz'
	scheme_nums=[alphabet.index(letter)+1 if scheme.count(letter)>1 else 0 for letter in scheme]
	return scheme_nums

def nums2scheme(nums):
	alphabet='-abcdefghijklmnopqrstuvwxyz'
	return [alphabet[n] if n<len(alphabet) else n for n in nums]

def transpose_up(slice):
	import string
	return ''.join(string.ascii_lowercase[sx-1] if sx else 'x' for sx in slice)


def schemenums2dict(scheme):
	d={}
	for i,x in enumerate(scheme):
		for ii,xx in enumerate(scheme[:i]):
			if x==xx:
				d[i]=ii
	return d




#####
# MISC FUNCTIONS
#####

def product(*args):
	if not args:
		return iter(((),)) # yield tuple()
	return (items + (item,)
		for items in product(*args[:-1]) for item in args[-1])

def read_tsv(fn,sep='\t'):
	import csv
	with codecs.open(fn,encoding='utf-8') as f:
		return list(csv.DictReader(f,delimiter=sep))

def hash(string):
	import hashlib
	if type(string)==unicode:
		string=string.encode('utf-8')
	return str(hashlib.sha224(string).hexdigest())

def toks2freq(l,tfy=False):
	c=Counter(l)
	if tfy:
		summ=float(sum(c.values()))
		for k,v in c.items():
			c[k]=v/summ
	return c



def slicex(l,num_slices=None,slice_length=None,runts=True,random=False):
	"""
	Returns a new list of n evenly-sized segments of the original list
	"""
	if random:
		import random
		random.shuffle(l)
	if not num_slices and not slice_length: return l
	if not slice_length: slice_length=int(len(l)/num_slices)
	newlist=[l[i:i+slice_length] for i in range(0, len(l), slice_length)]
	if runts: return newlist
	return [lx for lx in newlist if len(lx)==slice_length]



## Constants

BEATNAMES = {
	1:'Monometer',2:'Dimeter',3:'Trimeter',4:'Tetrameter',
	5:'Pentameter',6:'Hexameter',7:'Heptameter',8:'Octameter',
	9: 'Enneameter', 10:'Decameter', 11:'Hendecameter',12:'Dodecameter'}
