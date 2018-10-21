#pragma once

namespace NNS
{
	class IBase {
	public:
		struct SDeleter 
		{
			void operator()(IBase* p) const 
			{ 
				if (p)
				{
					p->Free();
				}
			}
		};
	public:
		virtual ~IBase() = default;
		virtual void Free() const = 0;
	};
}