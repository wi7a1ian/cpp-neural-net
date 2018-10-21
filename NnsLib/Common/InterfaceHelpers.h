#include <memory>

template<typename Base, typename T>
inline Base* as(T* obj) {
	return dynamic_cast<Base*>(obj);
}

template<typename To, typename From>
inline To* as(From& obj) {
	return dynamic_cast<To*>(&obj);
}

template<typename To, typename From, typename... FTypes>
inline To* as(std::unique_ptr<From, FTypes...>& ptr) {
	return dynamic_cast<To*>(ptr.get());
}

template<typename To, typename From, typename... FTypes>
inline std::unique_ptr<To, FTypes...> as_ptr(std::unique_ptr<From, FTypes...>& ptr) {
	std::unique_ptr<To, FTypes...> out_ptr;
	if (auto raw_ptr = dynamic_cast<To*>(ptr.get()))
	{
		ptr.release();
		out_ptr.reset(raw_ptr);
	}
	return out_ptr;
}

template<typename To, typename From, typename... FTypes>
inline std::unique_ptr<To, FTypes...> as_ptr(std::unique_ptr<From, FTypes...>&& ptr) {
	std::unique_ptr<To, FTypes...> out_ptr;
	if (auto raw_ptr = dynamic_cast<To*>(ptr.get()))
	{
		ptr.release();
		out_ptr.reset(raw_ptr);
	}
	return out_ptr;
}
